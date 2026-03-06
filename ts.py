import cshogi
import numpy as np

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
        return piece_map[raw_pt] * 81 + rotated_to
    else:
        az_from = cshogi_to_az_sq(cshogi.move_from(move))
        az_to = cshogi_to_az_sq(cshogi.move_to(move))
        rotated_from = (80 - az_from) if is_white_move else az_from
        rotated_to = (80 - az_to) if is_white_move else az_to
        rf_r, rf_f = divmod(rotated_from, 9)
        rt_r, rt_f = divmod(rotated_to, 9)
        dx, dy = rt_f - rf_f, rt_r - rf_r
        move_id = map_dx_dy_to_direction_id(dx, dy)
        return (7 * 81) + (rotated_from * 132) + (move_id * 2) + (1 if cshogi.move_is_promotion(move) else 0)

# --- 検証実行 ---
board = cshogi.Board()

# 1. 先手 7g7f
# board.move_from_usi (インスタンスメソッド) を使う
move_1 = board.move_from_usi("7g7f") 
print(f"Test 1: Sente 7g7f -> ActionID: {convert_move_to_az_id(move_1, board)}")
board.push(move_1) # 次の検証のために盤面を進める

# 2. 後手 3c3d (現在 board.turn は WHITE)
move_2 = board.move_from_usi("3c3d") 
print(f"Test 2: Gote 3c3d -> ActionID: {convert_move_to_az_id(move_2, board)}")