import os
import numpy as np
import cshogi
import re
import random
import shutil
import gc
from tqdm import tqdm

CSA_DIR = "/media/katiogoto/abb1e836-e529-4059-80be-167839ba757a/wdoor2025/2025"
TEMP_POOL_DIR = "/media/katiogoto/abb1e836-e529-4059-80be-167839ba757a/random_temp_pool"

OUTPUT_DIR = "/media/katiogoto/abb1e836-e529-4059-80be-167839ba757a/shogi_training_data/random_data"

# 100万件程度にする場合はここを 1000000 に
MAX_SAMPLES = 800000 
BATCH_SIZE = 10000 # 一時保存する単位

# 定数
K_SHOGI_BOARD_AREA = 81
K_SHOGI_POLICY_SIZE = 11259

def cshogi_to_az_sq(cshogi_sq):
    f = cshogi_sq // 9
    r = cshogi_sq % 9
    return r * 9 + f

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
            b = history_boards[-(t + 1)]
            rep = repetition_counts[-(t + 1)]
        else:
            b = cshogi.Board(); rep = 0
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

# =========================================================
#  メイン処理: ランダムデータ生成ロジック
# =========================================================

def process_random_rank():
    # ディレクトリ初期化
    if os.path.exists(TEMP_POOL_DIR): shutil.rmtree(TEMP_POOL_DIR)
    os.makedirs(TEMP_POOL_DIR, exist_ok=True)
    
    files = [f for f in os.listdir(CSA_DIR) if f.endswith(".csa")]
    random.shuffle(files)
    
    features_batch = []
    policy_batch = []
    value_batch = []
    
    total_count = 0
    batch_idx = 0
    
    # --- Phase 1: 収集と分割保存 (HDD) ---
    pbar = tqdm(total=MAX_SAMPLES, desc="Collecting Data to HDD")
    for filename in files:
        if total_count >= MAX_SAMPLES: break
        
        path = os.path.join(CSA_DIR, filename)
        board = cshogi.Board()
        history_boards, repetition_counts, board_hash_history = [], [], {}
        move_count = 0
        
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            
            # 手順開始位置の特定
            start_idx = 0
            for i, line in enumerate(lines):
                if line.strip() in ["+", "-"]:
                    board.turn = cshogi.BLACK if line.strip() == "+" else cshogi.WHITE
                    start_idx = i + 1
                    break
            
            for line in lines[start_idx:]:
                if total_count >= MAX_SAMPLES: break
                line = line.strip()
                if not (line.startswith("+") or line.startswith("-")): continue
                
                # 特徴量作成
                curr_board = board.copy()
                history_boards.append(curr_board)
                h = hash(board.sfen())
                board_hash_history[h] = board_hash_history.get(h, 0) + 1
                repetition_counts.append(min(board_hash_history[h], 3))
                
                feat = board_to_tensor_alpha_zero(history_boards, repetition_counts, move_count, board.turn)
                
                # ランダムPolicy
                legal_moves = list(board.legal_moves)
                if not legal_moves: break
                az_id = convert_move_to_az_id(random.choice(legal_moves), board)
                
                if az_id != -1:
                    pol = np.zeros(K_SHOGI_POLICY_SIZE, dtype=np.float32)
                    pol[az_id] = 1.0
                    
                    features_batch.append(feat)
                    policy_batch.append(pol)
                    value_batch.append(0.0)
                    total_count += 1
                    pbar.update(1)
                
                # バッチサイズに達したらHDDへ保存
                if len(features_batch) >= BATCH_SIZE:
                    save_batch(batch_idx, features_batch, policy_batch, value_batch)
                    features_batch, policy_batch, value_batch = [], [], []
                    batch_idx += 1
                    gc.collect() # メモリ解放
                
                move_str = line[1:]
                try:
                    board.push_csa(move_str)
                    move_count += 1
                except: break
        except: continue
    
    # 残りのデータを保存
    if features_batch:
        save_batch(batch_idx, features_batch, policy_batch, value_batch)
    pbar.close()

    # --- Phase 2: シャッフルしてマージ (HDD -> SSD) ---
    finalize_and_merge(total_count)

def save_batch(idx, f, p, v):
    """HDDに一時保存"""
    np.save(os.path.join(TEMP_POOL_DIR, f"feat_{idx}.npy"), np.array(f, dtype=np.float32))
    np.save(os.path.join(TEMP_POOL_DIR, f"pol_{idx}.npy"), np.array(p, dtype=np.float32))
    np.save(os.path.join(TEMP_POOL_DIR, f"val_{idx}.npy"), np.array(v, dtype=np.float32))

def finalize_and_merge(total_count):
    """HDDから読み込み、局面単位でシャッフルしてHDDに保存 (シーケンシャル書き込み)"""
    print(f"\nPhase 2: Merging and Shuffling {total_count} samples (Optimized for HDD)...")
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 書き込み先インデックスをシャッフルするのではなく、
    # 「どのソースファイルのどの行を、順番に 0, 1, 2... と書き込むか」のポインタを作る
    batch_files = sorted([f for f in os.listdir(TEMP_POOL_DIR) if f.startswith("feat_")])
    
    all_pointers = []
    for bf in batch_files:
        idx_str = bf.split('_')[1].split('.')[0]
        num_in_file = len(np.load(os.path.join(TEMP_POOL_DIR, bf), mmap_mode='r'))
        for i in range(num_in_file):
            all_pointers.append((idx_str, i))
    
    random.shuffle(all_pointers) # ここで局面単位のシャッフルを完結

    # 出力用 memmap (HDD)
    f_map = np.lib.format.open_memmap(os.path.join(OUTPUT_DIR, "features.npy"), mode='w+', dtype=np.float32, shape=(total_count, 362, 9, 9))
    p_map = np.lib.format.open_memmap(os.path.join(OUTPUT_DIR, "policy.npy"), mode='w+', dtype=np.float32, shape=(total_count, K_SHOGI_POLICY_SIZE))
    v_map = np.lib.format.open_memmap(os.path.join(OUTPUT_DIR, "value.npy"), mode='w+', dtype=np.float32, shape=(total_count,))
    # BT学習用のrank.npyも作成（r_infinityなので値は0.0）
    r_map = np.lib.format.open_memmap(os.path.join(OUTPUT_DIR, "rank.npy"), mode='w+', dtype=np.float32, shape=(total_count, 1))

    # --- シーケンシャル書き込み実行 ---
    WRITE_BATCH = 5000 
    for i in tqdm(range(0, total_count, WRITE_BATCH), desc="Sequential Writing to HDD"):
        batch_ptr = all_pointers[i : i + WRITE_BATCH]
        curr_size = len(batch_ptr)
        
        f_buf = np.zeros((curr_size, 362, 9, 9), dtype=np.float32)
        p_buf = np.zeros((curr_size, K_SHOGI_POLICY_SIZE), dtype=np.float32)
        v_buf = np.zeros(curr_size, dtype=np.float32)

        unique_file_indices = set(p[0] for p in batch_ptr)
        for f_idx in unique_file_indices:
            tmp_f = np.load(os.path.join(TEMP_POOL_DIR, f"feat_{f_idx}.npy"), mmap_mode='r')
            tmp_p = np.load(os.path.join(TEMP_POOL_DIR, f"pol_{f_idx}.npy"), mmap_mode='r')
            tmp_v = np.load(os.path.join(TEMP_POOL_DIR, f"val_{f_idx}.npy"), mmap_mode='r')
            
            for b_idx, (p_f_idx, p_row) in enumerate(batch_ptr):
                if p_f_idx == f_idx:
                    f_buf[b_idx] = tmp_f[p_row]
                    p_buf[b_idx] = tmp_p[p_row]
                    v_buf[b_idx] = tmp_v[p_row]
        
        # HDDへ連続書き込み
        f_map[i : i + curr_size] = f_buf
        p_map[i : i + curr_size] = p_buf
        v_map[i : i + curr_size] = v_buf
        r_map[i : i + curr_size] = 0.0

    f_map.flush(); p_map.flush(); v_map.flush(); r_map.flush()
    shutil.rmtree(TEMP_POOL_DIR)
    print("Done.")

if __name__ == "__main__":
    process_random_rank()