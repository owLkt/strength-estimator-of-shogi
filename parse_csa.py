import os
import numpy as np
import cshogi
from tqdm import tqdm
import re
import random
import gc
import shutil
import csv
import sys

# ---------------------------------------------------------
# パス・設定値
# ---------------------------------------------------------
# base_dir は「実行場所」に左右されず、常にスクリプトの横に保存するために使用します
base_dir = os.path.dirname(os.path.abspath(__file__))

# HDD上の一時保存ディレクトリ
temp_pool_dir = "/media/katiogoto/abb1e836-e529-4059-80be-167839ba757a/rank_pool_temp"
final_data_root = "/media/katiogoto/abb1e836-e529-4059-80be-167839ba757a/shogi_training_data"

min_elo = 1500
max_elo = 4700
interval = 400
COLLECT_LIMIT_TOTAL = 1000000
RATIO_CAND = 0.01
RATIO_TEST = 0.1
RATIO_TRAIN = 0.89

num_rank = (max_elo - min_elo) // interval
K_SHOGI_BOARD_AREA = 81
K_SHOGI_POLICY_SIZE = 11259

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

fix_seed(42)

# ---------------------------------------------------------
# 安全な削除関数 (シンボリックリンク対策)
# ---------------------------------------------------------
def safe_remove(path):
    if os.path.lexists(path):
        if os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

# ---------------------------------------------------------
# 座標・特徴量変換ロジック
# ---------------------------------------------------------
def cshogi_to_az_sq(cshogi_sq):
    return (cshogi_sq % 9) * 9 + (cshogi_sq // 9)

def map_dx_dy_to_direction_id(dx, dy):
    if dx == -1 and dy == -2: return 0
    if dx == 1 and dy == -2: return 1
    if dx == 0 and dy < 0: return 2 + (-dy - 1)
    if dx == 0 and dy > 0: return 10 + (dy - 1)
    if dy == 0 and dx < 0: return 18 + (-dx - 1)
    if dy == 0 and dx > 0: return 26 + (dx - 1)
    if dx < 0 and dy < 0 and dx == dy: return 34 + (-dx - 1)
    if dx > 0 and dy < 0 and dx == -dy: return 42 + (dx - 1)
    if dx < 0 and dy > 0 and dx == -dy: return 50 + (-dx - 1)
    if dx > 0 and dy > 0 and dx == dy: return 58 + (dx - 1)
    return -1

def convert_move_to_az_id(move, board):
    is_white_move = (board.turn == cshogi.WHITE)
    if cshogi.move_is_drop(move):
        piece_map = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6}
        raw_pt = cshogi.move_drop_hand_piece(move)
        az_to = cshogi_to_az_sq(cshogi.move_to(move))
        rotated_to = (80 - az_to) if is_white_move else az_to
        return piece_map[raw_pt] * K_SHOGI_BOARD_AREA + rotated_to
    else:
        az_from = cshogi_to_az_sq(cshogi.move_from(move))
        az_to = cshogi_to_az_sq(cshogi.move_to(move))
        promote = cshogi.move_is_promotion(move)
        rotated_from = (80 - az_from) if is_white_move else az_from
        rotated_to = (80 - az_to) if is_white_move else az_to
        rf_r, rf_f = divmod(rotated_from, 9)
        rt_r, rt_f = divmod(rotated_to, 9)
        dx, dy = rt_f - rf_f, rt_r - rf_r
        move_id = map_dx_dy_to_direction_id(dx, dy)
        if move_id == -1: return -1
        return (7 * K_SHOGI_BOARD_AREA) + (rotated_from * 132) + (move_id * 2) + (1 if promote else 0)

def piece_planes(board, is_black, flip):
    planes = np.zeros((9, 9, 14), dtype=np.float32)
    mapping = {1:0, 2:1, 3:2, 4:3, 7:4, 5:5, 6:6, 8:7, 9:8, 10:9, 11:10, 12:11, 13:12, 14:13}
    for sq in range(81):
        piece = board.pieces[sq]
        if piece == 0 or (piece <= 14) != is_black: continue
        az_sq = cshogi_to_az_sq(sq)
        r, f = divmod(az_sq, 9)
        if flip: r, f = 8 - r, 8 - f
        raw_type = piece if piece <= 14 else (piece - 16)
        if raw_type in mapping: planes[r, f, mapping[raw_type]] = 1.0
    return planes

def prisoner_planes(board, is_black):
    hand = board.pieces_in_hand[0 if is_black else 1]
    ordered_hand = np.array([hand[0], hand[1], hand[2], hand[3], hand[5], hand[6], hand[4]], dtype=np.float32)
    return np.tile(ordered_hand, (9, 9, 1))

def board_to_tensor_alpha_zero(history_boards, repetition_counts, move_count, turn):
    T = 8
    planes = []
    flip = (turn == cshogi.WHITE)
    us_black = (turn == cshogi.BLACK)
    for t in range(T):
        if t < len(history_boards):
            b, rep = history_boards[-(t + 1)], repetition_counts[-(t + 1)]
        else:
            b, rep = cshogi.Board(), 0
        planes.append(piece_planes(b, us_black, flip))
        planes.append(piece_planes(b, not us_black, flip))
        rep_p = np.zeros((9, 9, 3), dtype=np.float32)
        if 1 <= rep <= 3: rep_p[:, :, rep - 1] = 1.0
        planes.append(rep_p)
        planes.append(prisoner_planes(b, us_black))
        planes.append(prisoner_planes(b, not us_black))
    turn_plane = np.full((9, 9, 1), 1.0 if turn == cshogi.BLACK else 0.0, dtype=np.float32)
    move_plane = np.full((9, 9, 1), move_count / 512.0, dtype=np.float32)
    return np.concatenate(planes + [turn_plane, move_plane], axis=2).transpose(2, 0, 1)

def getRankIndex(elo):
    if elo < min_elo or elo >= max_elo: return -1
    return int((elo - min_elo) // interval)

# ---------------------------------------------------------
# 最適化されたパース関数
# ---------------------------------------------------------
def parse_kifu_and_features(file_path, collected_counts, limit):
    board = cshogi.Board()
    move_count, history_boards, repetition_counts, board_hash_history = 0, [], [], {}
    black_rate, white_rate, winner = None, None, None
    ranked_samples = {r: [] for r in range(num_rank)}
    
    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        
        # 1. Eloレートの早期取得
        for line in lines:
            line = line.strip()
            if line.startswith("'black_rate:"):
                m = re.search(r"'black_rate:[^:]+:(\d+)", line)
                if m: black_rate = int(m.group(1))
            elif line.startswith("'white_rate:"):
                m = re.search(r"'white_rate:[^:]+:(\d+)", line)
                if m: white_rate = int(m.group(1))
            if black_rate and white_rate: break
        
        if black_rate is None or white_rate is None: return {}

        r_idx_black = getRankIndex(black_rate)
        r_idx_white = getRankIndex(white_rate)
        
        # 2. 双方のElo帯が上限に達しているなら、このファイルは即終了
        can_black = (r_idx_black != -1 and collected_counts[r_idx_black] < limit)
        can_white = (r_idx_white != -1 and collected_counts[r_idx_white] < limit)
        if not can_black and not can_white:
            return {}

        found_moves, temp_data = False, []
        for line in lines:
            line = line.strip()
            if not found_moves:
                if line == "+": board.turn = cshogi.BLACK; found_moves = True
                elif line == "-": board.turn = cshogi.WHITE; found_moves = True
                continue
            
            if line.startswith("+") or line.startswith("-"):
                move_str = line[1:]
                if len(move_str) < 6 or not move_str[0:4].isdigit(): continue
                
                curr_board = board.copy()
                history_boards.append(curr_board)
                h = hash(board.sfen())
                board_hash_history[h] = board_hash_history.get(h, 0) + 1
                repetition_counts.append(min(board_hash_history[h], 3))
                
                target_player = board.turn
                player_rate = black_rate if target_player == cshogi.BLACK else white_rate
                r_idx = getRankIndex(player_rate)
                
                # 3. 収集対象かつ上限未満の場合のみ、重い特徴量生成を実行
                feat = None
                if r_idx != -1 and collected_counts[r_idx] < limit:
                    feat = board_to_tensor_alpha_zero(history_boards, repetition_counts, move_count, target_player)
                
                try:
                    move = board.push_csa(move_str)
                    if feat is not None:
                        az_id = convert_move_to_az_id(move, curr_board)
                        if az_id != -1:
                            pol = np.zeros(K_SHOGI_POLICY_SIZE, dtype=np.float32)
                            pol[az_id] = 1.0
                            temp_data.append({'r_idx': r_idx, 'f': feat, 'p': pol, 'turn': target_player})
                    move_count += 1
                except: continue
            elif line.startswith("%"):
                if "%TORYO" in line or "%TIME_UP" in line: winner = 1 - board.turn
                break
        
        for item in temp_data:
            val = 0.0
            if winner is not None: val = 1.0 if item['turn'] == winner else -1.0
            ranked_samples[item['r_idx']].append((item['f'], item['p'], val))
            
    except: return {}
    return ranked_samples

def getRankIndex(elo):
    if elo < min_elo or elo >= max_elo: return -1
    return int((elo - min_elo) // interval)

# ---------------------------------------------------------
# Phase 1: 保存関数 (HDD への書き出し)
# ---------------------------------------------------------
def save_to_pool(rank_bins, batch_idx):
    """バッチごとにHDDへ一時保存。メモリ解放を促進。"""
    for r_idx, games in enumerate(rank_bins):
        if not games: continue
        f_list, p_list, v_list, l_list = [], [], [], []
        for g in games:
            l_list.append(len(g))
            for s in g:
                f_list.append(s[0]); p_list.append(s[1]); v_list.append(s[2])
        
        out_dir = os.path.join(temp_pool_dir, f"rank_{r_idx}")
        os.makedirs(out_dir, exist_ok=True)
        suffix = f"_batch_{batch_idx}"
        
        # HDDへ保存
        np.save(os.path.join(out_dir, f"feat{suffix}.npy"), np.array(f_list, dtype=np.float32))
        np.save(os.path.join(out_dir, f"pol{suffix}.npy"), np.array(p_list, dtype=np.float32))
        np.save(os.path.join(out_dir, f"val{suffix}.npy"), np.array(v_list, dtype=np.float32))
        np.save(os.path.join(out_dir, f"len{suffix}.npy"), np.array(l_list, dtype=np.int32))
    gc.collect()

# ---------------------------------------------------------
# Phase 2: マージ・シャッフル関数 (HDD)
# ---------------------------------------------------------
def finalize_and_split_data():
    print(f"\n[Phase 2] Shuffling and merging (Optimized for HDD)...")
    final_roots = {
        "train": os.path.join(final_data_root, "training_shogi"),
        "test": os.path.join(final_data_root, "query_shogi"),
        "cand": os.path.join(final_data_root, "candidate_shogi")
    }
    
    for r in final_roots.values(): 
        safe_remove(r)
        os.makedirs(r, exist_ok=True)

    for r_idx in range(num_rank):
        source_dir = os.path.join(temp_pool_dir, f"rank_{r_idx}")
        if not os.path.isdir(source_dir): continue

        # --- 1. メタデータの準備 ---
        batch_files = sorted([f for f in os.listdir(source_dir) if f.startswith("len")])
        all_game_refs = [] 
        for bf in batch_files:
            suffix = bf.replace("len", "").replace(".npy", "")
            lengths = np.load(os.path.join(source_dir, bf))
            start_pos = 0
            for length in lengths:
                all_game_refs.append({"suffix": suffix, "start": start_pos, "len": length})
                start_pos += length

        # --- 2. 分割 (ゲーム単位) ---
        random.shuffle(all_game_refs)
        total_g = len(all_game_refs)
        idx_test = int(total_g * RATIO_TRAIN)
        idx_cand = idx_test + int(total_g * RATIO_TEST)
        assigned = {
            "train": all_game_refs[:idx_test],
            "test": all_game_refs[idx_test:idx_cand],
            "cand": all_game_refs[idx_cand:]
        }

        lo, hi = min_elo + interval * r_idx, min_elo + interval * (r_idx + 1)
        rank_label = np.zeros(num_rank, dtype=np.float32); rank_label[r_idx] = 1.0

        for mode, games in assigned.items():
            if not games: continue
            out_dir = os.path.join(final_roots[mode], f"{lo}_{hi}")
            os.makedirs(out_dir, exist_ok=True)
            total_pos = int(sum(g['len'] for g in games))
            
            # 出力用 memmap (HDD上の最終保存先)
            f_map = np.lib.format.open_memmap(os.path.join(out_dir, "features.npy"), mode='w+', dtype=np.float32, shape=(total_pos, 362, 9, 9))
            p_map = np.lib.format.open_memmap(os.path.join(out_dir, "policy.npy"), mode='w+', dtype=np.float32, shape=(total_pos, K_SHOGI_POLICY_SIZE))
            v_map = np.lib.format.open_memmap(os.path.join(out_dir, "value.npy"), mode='w+', dtype=np.float32, shape=(total_pos,))
            r_map = np.lib.format.open_memmap(os.path.join(out_dir, "rank.npy"), mode='w+', dtype=np.float32, shape=(total_pos, num_rank))
            
            # --- 3. 書き込み順序リストの作成 ---
            pos_pointers = []
            if mode == "train":
                # Trainは局面単位でさらにシャッフル
                for g in games:
                    for offset in range(g['len']):
                        pos_pointers.append((g['suffix'], g['start'] + offset))
                random.shuffle(pos_pointers)
            else:
                # Test/Cand は元のゲーム順を維持
                for g in games:
                    for offset in range(g['len']):
                        pos_pointers.append((g['suffix'], g['start'] + offset))

            # --- 4. バッファリング書き込み (シーケンシャル処理) ---
            WRITE_BATCH = 4000 # メモリと相談して調整
            for i in tqdm(range(0, len(pos_pointers), WRITE_BATCH), desc=f"{mode} Rank {r_idx}"):
                batch = pos_pointers[i : i + WRITE_BATCH]
                curr_batch_size = len(batch)
                
                # RAM上にバッファを確保
                f_buf = np.zeros((curr_batch_size, 362, 9, 9), dtype=np.float32)
                p_buf = np.zeros((curr_batch_size, K_SHOGI_POLICY_SIZE), dtype=np.float32)
                v_buf = np.zeros(curr_batch_size, dtype=np.float32)

                # 必要なソースファイルをまとめて読み込み
                unique_sufs = set(p[0] for p in batch)
                for suf in unique_sufs:
                    tmp_f = np.load(os.path.join(source_dir, f"feat{suf}.npy"), mmap_mode='r')
                    tmp_p = np.load(os.path.join(source_dir, f"pol{suf}.npy"), mmap_mode='r')
                    tmp_v = np.load(os.path.join(source_dir, f"val{suf}.npy"), mmap_mode='r')
                    
                    for b_idx, (p_suf, p_src) in enumerate(batch):
                        if p_suf == suf:
                            f_buf[b_idx] = tmp_f[p_src]
                            p_buf[b_idx] = tmp_p[p_src]
                            v_buf[b_idx] = tmp_v[p_src]
                
                # 書き込み先は常に i : i + curr_batch_size (順番通り)
                f_map[i : i + curr_batch_size] = f_buf
                p_map[i : i + curr_batch_size] = p_buf
                v_map[i : i + curr_batch_size] = v_buf
                r_map[i : i + curr_batch_size] = rank_label

            if mode != "train":
                # Test/Cand の場合は元の仕様通り game_lengths を保存
                game_lengths = [g['len'] for g in games]
                np.save(os.path.join(out_dir, "game_lengths.npy"), np.array(game_lengths, dtype=np.int32))

            f_map.flush(); p_map.flush(); v_map.flush(); r_map.flush()
            del f_map, p_map, v_map, r_map; gc.collect()

# ---------------------------------------------------------
# main 処理
# ---------------------------------------------------------
def main():
    csa_dirs = [
        "/media/katiogoto/abb1e836-e529-4059-80be-167839ba757a/wdoor2024/2024",
        "/media/katiogoto/abb1e836-e529-4059-80be-167839ba757a/wdoor2023/2023",
        "/media/katiogoto/abb1e836-e529-4059-80be-167839ba757a/wdoor2023/2025"
    ]
    files = []
    for d in csa_dirs:
        if not os.path.isdir(d): continue
        files.extend([os.path.join(d, f) for f in os.listdir(d) if f.endswith(".csa")])
    random.shuffle(files)

    safe_remove(temp_pool_dir)
    os.makedirs(temp_pool_dir, exist_ok=True)

    file_stats = {}
    collected, batch_counter = [0] * num_rank, 0
    rank_bins = [[] for _ in range(num_rank)]

    print(f"Phase 1: Collecting to HDD. Limit per Rank: {COLLECT_LIMIT_TOTAL}")

    for i, file_path in enumerate(tqdm(files, desc="Processing")):
        # 現在の収集状況(collected)を渡して、不要な処理をスキップさせる
        ranked_samples = parse_kifu_and_features(file_path, collected, COLLECT_LIMIT_TOTAL)
        
        total_pos_in_file = 0
        if ranked_samples:
            for r_idx, samples in ranked_samples.items():
                if samples:
                    num_samples = len(samples)
                    rank_bins[r_idx].append(samples)
                    collected[r_idx] += num_samples
                    total_pos_in_file += num_samples
        
        file_stats[file_path] = total_pos_in_file
        
        # バッチ保存
        if (i + 1) % 500 == 0:
            # すべてのランクが上限に達したかチェック
            if all(c >= COLLECT_LIMIT_TOTAL for c in collected):
                print("\nAll limits reached. Stopping collection early.")
                break
                
            save_to_pool(rank_bins, batch_counter)
            rank_bins = [[] for _ in range(num_rank)]
            batch_counter += 1
            gc.collect()

    save_to_pool(rank_bins, batch_counter)

    # 統計出力 (SSD)
    stats_csv = os.path.join(base_dir, "extraction_stats.csv")
    with open(stats_csv, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f); writer.writerow(["file_path", "positions_extracted"])
        for path, count in file_stats.items(): writer.writerow([path, count])

    # Phase 2: HDD -> SSD
    from __main__ import finalize_and_split_data # スクリプト内の関数呼び出し
    finalize_and_split_data()

    safe_remove(temp_pool_dir)
    print("\nProcess finished.")

if __name__ == "__main__":
    main()