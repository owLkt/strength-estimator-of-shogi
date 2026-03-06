import numpy as np
import os
import cshogi
from tqdm import tqdm

# --- 設定（作成側と完全に一致させる） ---
# Sunfishの駒順: 歩, 香, 桂, 銀, 金, 角, 飛, 玉 + 成駒
PIECE_NAMES = ["歩", "香", "桂", "銀", "金", "角", "飛", "玉", "と", "杏", "圭", "全", "馬", "龍"]
# Sunfishの持ち駒順: P, L, N, S, B, R, G (1~7)
HAND_NAMES  = ["歩", "香", "桂", "銀", "角", "飛", "金"]

def get_to_sq_from_direction(from_sq, direction_id):
    """Rank-Major座標系での移動先計算 (C++: get_to_sq_from_direction 互換)"""
    from_rank, from_file = divmod(from_sq, 9)
    dx, dy = 0, 0
    
    if direction_id == 0: dx, dy = -1, -2  # 桂馬(左)
    elif direction_id == 1: dx, dy = 1, -2  # 桂馬(右)
    elif 2 <= direction_id <= 9:   dy = -(direction_id - 2 + 1)   # 北
    elif 10 <= direction_id <= 17: dy = (direction_id - 10 + 1)   # 南
    elif 18 <= direction_id <= 25: dx = -(direction_id - 18 + 1)  # 西
    elif 26 <= direction_id <= 33: dx = (direction_id - 26 + 1)   # 東
    elif 34 <= direction_id <= 41:
        dist = direction_id - 34 + 1; dx, dy = -dist, -dist       # 北西
    elif 42 <= direction_id <= 49:
        dist = direction_id - 42 + 1; dx, dy = dist, -dist        # 北東
    elif 50 <= direction_id <= 57:
        dist = direction_id - 50 + 1; dx, dy = -dist, dist        # 南西
    elif 58 <= direction_id <= 65:
        dist = direction_id - 58 + 1; dx, dy = dist, dist         # 南東
    else: return -1
    
    to_rank, to_file = from_rank + dy, from_file + dx
    if 0 <= to_rank <= 8 and 0 <= to_file <= 8:
        return to_rank * 9 + to_file
    return -1

def action_id_to_usi(az_id):
    """AZ IDを人間が読める形式に変換"""
    BOARD_AREA = 81
    if az_id < 7 * BOARD_AREA:
        piece_type = az_id // BOARD_AREA
        to_sq = az_id % BOARD_AREA
        tr, tf = divmod(to_sq, 9)
        return f"{HAND_NAMES[piece_type]}*{(9-tf)}{chr(ord('a')+tr)}"
    else:
        move_id = az_id - (7 * BOARD_AREA)
        from_sq = move_id // 132
        type_id = move_id % 132
        dir_id, promote = divmod(type_id, 2)
        to_sq = get_to_sq_from_direction(from_sq, dir_id)
        fr, ff = divmod(from_sq, 9)
        if to_sq != -1:
            tr, tf = divmod(to_sq, 9)
            to_str = f"{(9-tf)}{chr(ord('a')+tr)}"
        else:
            to_str = "ERR"
        promote_str = "+" if promote else ""
        return f"{(9-ff)}{chr(ord('a')+fr)}{to_str}{promote_str}"

def run_deep_health_check(features, policy, value, rank, sample_size=10000):
    total = min(len(features), sample_size)
    errors = {
        "move_origin_empty": 0,
        "move_dest_self_occupied": 0,
        "move_dest_out_of_range": 0,
        "drop_on_piece": 0,
        "nifu_violation": 0,
        "policy_not_one_hot": 0,
        "invalid_value_range": 0,
        "piece_count_overflow": 0
    }
    
    print(f"\n[Deep Check] Scanning {total} samples...")
    
    for i in tqdm(range(total)):
        f, p, v, r = features[i], policy[i], value[i], rank[i]
        
        # PolicyのOne-hotチェック (和がほぼ1か)
        if not (0.9 <= np.sum(p) <= 1.1):
            errors["policy_not_one_hot"] += 1
        
        argmax_idx = np.argmax(p)
        us_board = np.sum(f[0:14], axis=0)
        enemy_board = np.sum(f[14:28], axis=0)
        
        if argmax_idx >= 567: # 移動
            move_data = argmax_idx - 567
            from_sq = move_data // 132
            fy, fx = divmod(from_sq, 9)
            dir_id = (move_data % 132) // 2
            to_sq = get_to_sq_from_direction(from_sq, dir_id)
            
            if to_sq == -1:
                errors["move_dest_out_of_range"] += 1
            else:
                ty, tx = divmod(to_sq, 9)
                if us_board[fy, fx] == 0:
                    errors["move_origin_empty"] += 1
                if us_board[ty, tx] > 0:
                    errors["move_dest_self_occupied"] += 1
        else: # 打ち駒
            piece_type = argmax_idx // 81
            to_sq = argmax_idx % 81
            ty, tx = divmod(to_sq, 9)
            if us_board[ty, tx] > 0 or enemy_board[ty, tx] > 0:
                errors["drop_on_piece"] += 1
            # 二歩チェック (Sunfish順: 歩=0)
            if piece_type == 0:
                if np.any(f[0, :, tx] == 1):
                    errors["nifu_violation"] += 1
        
        if abs(v) > 1.0: errors["invalid_value_range"] += 1
        if np.sum(us_board + enemy_board) > 40: errors["piece_count_overflow"] += 1

    print("\n" + "="*45)
    print(f"{'CHECK ITEM':<25} | {'COUNT':<7} | {'STATUS'}")
    print("-" * 45)
    for k, count in errors.items():
        status = "OK" if count == 0 else "!! ERR !!"
        print(f"{k:<25} | {count:<7} | {status}")
    print("="*45)

def visualize_detailed_sample(features, policy, value, rank):
    argmax_idx = np.argmax(policy)
    usi_move = action_id_to_usi(argmax_idx)
    
    # 手番(360ch)と手数(361ch) ※末尾の平面
    turn = "Black" if features[360, 0, 0] == 1.0 else "White"
    move_num = int(features[361, 0, 0] * 512)
    
    print(f"\n[Detail View] Move: {usi_move} | Val: {value:.2f} | Turn: {turn} | Move#: {move_num}")

    print("\n   9  8  7  6  5  4  3  2  1")
    for y in range(9):
        row = f"{chr(ord('a')+y)} "
        for x in range(9):
            p_char = " . "
            for p in range(14):
                if features[p, y, x] == 1: p_char = f" {PIECE_NAMES[p]} "
            for p in range(14):
                if features[14+p, y, x] == 1: p_char = f"v{PIECE_NAMES[p]}"
            
            # 移動元を()で囲む
            if argmax_idx >= 567 and (y*9+x) == (argmax_idx - 567)//132:
                row += f"({p_char.strip()})"
            else:
                row += p_char
        print(row)
    
    # 持ち駒 (31-37ch: Us, 38-44ch: Enemy)
    s_hand = [f"{HAND_NAMES[i]}x{int(features[31+i,0,0])}" for i in range(7) if features[31+i,0,0] > 0]
    e_hand = [f"{HAND_NAMES[i]}x{int(features[38+i,0,0])}" for i in range(7) if features[38+i,0,0] > 0]
    print(f"Hand(Us): {' '.join(s_hand)}")
    print(f"Hand(En): {' '.join(e_hand)}")
    print("-" * 45)

if __name__ == "__main__":
    target_dir = "training_shogi/3300_3500" # 環境に合わせて変更
    if os.path.exists(f"{target_dir}/features.npy"):
        f = np.load(f"{target_dir}/features.npy", mmap_mode='r')
        p = np.load(f"{target_dir}/policy.npy", mmap_mode='r')
        v = np.load(f"{target_dir}/value.npy", mmap_mode='r')
        r = np.load(f"{target_dir}/rank.npy", mmap_mode='r')
        
        run_deep_health_check(f, p, v, r)
        
        # 3件ランダムに表示
        indices = np.random.choice(len(f), 3, replace=False)
        for idx in indices:
            visualize_detailed_sample(f[idx], p[idx], v[idx], r[idx])
            input("\nNext -> Press Enter")