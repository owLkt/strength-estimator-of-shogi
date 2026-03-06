#include <iostream>
#include <string>
#include <cassert> // assert() を使うために必要
#include "move.h"
#include "Piece.h"
#include "Square.h"

// 名前空間は convertAZ などのロジックを囲む
namespace minizero::env::shogi {

const int kShogiBoardSize = 9;
const int kShogiBoardArea = kShogiBoardSize * kShogiBoardSize;
// 駒打ちで使うIDの総数（7種類の駒 * 81マス）
const int kNumDropMoves = 7 * (kShogiBoardSize * kShogiBoardSize); // 567

// --- ↓↓↓ テスト対象の関数 ↓↓↓ ---
    static int map_dx_dy_to_direction_id(int dx, int dy) {
        // 座標系 (0,0) = 9一 (左上) と仮定。
        // dy < 0 は「上」方向、dx < 0 は「左」方向。

        // 桂馬の動き
        if (dx == -1 && dy == -2) return 0; // 桂馬 (左)
        if (dx == 1 && dy == -2) return 1;  // 桂馬 (右)

        // 直線・斜めの動き
        int distance = 0;

        if (dx == 0 && dy < 0) { // 上
            distance = -dy; // 1-8
            return 2 + (distance - 1); // ID 2-9
        }
        if (dx == 0 && dy > 0) { // 下
            distance = dy;
            return 10 + (distance - 1); // ID 10-17
        }
        if (dy == 0 && dx < 0) { // 左
            distance = -dx;
            return 18 + (distance - 1); // ID 18-25
        }
        if (dy == 0 && dx > 0) { // 右
            distance = dx;
            return 26 + (distance - 1); // ID 26-33
        }

        if (dx < 0 && dy < 0 && dx == dy) { // 左上
            distance = -dx;
            return 34 + (distance - 1); // ID 34-41
        }
        if (dx > 0 && dy < 0 && dx == -dy) { // 右上
            distance = dx;
            return 42 + (distance - 1); // ID 42-49
        }
        if (dx < 0 && dy > 0 && dx == -dy) { // 左下
            distance = -dx;
            return 50 + (distance - 1); // ID 50-57
        }
        if (dx > 0 && dy > 0 && dx == dy) { // 右下
            distance = dx;
            return 58 + (distance - 1); // ID 58-65
        }

        // 該当しない（不正な動き）
        return -1; // エラー
    }

    /**
     * from_sq (0-80) と to_sq (0-80) から、
     * 相対的な方向ID (0-65) を計算
     * (sq % 9 が X座標, sq / 9 が Y座標)
     * @param from_sq 移動元のマスID (0-80)
     * @param to_sq   移動先のマスID (0-80)
     * @return int 方向ID (0-65)。
     */
    static int get_direction_id(int from_sq, int to_sq) {
        int from_x = from_sq % 9;
        int from_y = from_sq / 9;
        int to_x = to_sq % 9;
        int to_y = to_sq / 9;

        int dx = to_x - from_x;
        int dy = to_y - from_y;

        // (dx, dy) から 66種類のID (0-65) にマッピングする
        return map_dx_dy_to_direction_id(dx, dy);
    }

    //sunfish3 -> AlphaZero の行動ID変換
    static int convertAZ(int sunfish_id) {
        int az_action_id = -1;
        Move move = Move::deserialize(sunfish_id);

        int from = move.from().index();
        int to = move.to().index();
        bool promote = move.promote();

        if (move.isHand()){
            // 駒打ち (ID: 0 〜 566)
            int piece_type_id = move.piece().index() & Piece::KindMask; // 駒の種類IDを取得

            az_action_id = piece_type_id * kShogiBoardArea + to;
        } else {
            // 盤上移動 (ID: 567 〜)
            int promote_offset = promote ? 1 : 0;
            int move_id = get_direction_id(from, to);
            az_action_id = 7 * kShogiBoardArea + from * move_id + promote_offset;
        }

        return az_action_id;
    };
// --- ↑↑↑ テスト対象の関数ここまで ---

} // namespace minizero::env::shogi
// --- ★★★ 名前空間はここで閉じる ★★★ ---


// --- メイン関数 (★グローバル空間に置く) ---
int main() {
    // 名前空間内のクラスや定数を使うために using
    using namespace minizero::env::shogi;
    
    std::cout << "テストを開始します..." << std::endl;

    // --- 期待値の計算 (Pawn=0, Bishop=5 と仮定) ---
    // ※ Piece::BPawn.type() == 0, Piece::BBishop.type() == 5 であると仮定
    // (このマッピングが違う場合は、 expected_az_id_X の値を要修正)
    int PAWN_HAND_IDX = 0;   // 仮定 (Piece::BPawn.type() の値)
    int BISHOP_HAND_IDX = 5; // 仮定 (Piece::BBishop.type() の値)

    int expected_az_id_1 = (PAWN_HAND_IDX * 81) + 40;  // 40
    int expected_az_id_2 = (BISHOP_HAND_IDX * 81) + 19; // 424
    int move_id_3 = (60 * 81 * 2) + (51 * 2) + 0;      // 9822
    int expected_az_id_3 = kNumDropMoves + move_id_3;  // 10389
    int move_id_4 = (60 * 81 * 2) + (51 * 2) + 1;      // 9823 (テスト4は 60->51 成り)
    int expected_az_id_4 = kNumDropMoves + move_id_4;  // 10390


    // --- テストケースの準備 ---

    std::cout << "[テスト1: 歩 打ち (5五)]" << std::endl;
    Move move1 = Move(Piece::BPawn, Square(40), true); 
    int sunfish_id_1 = Move::serialize(move1); 
    
    std::cout << "[テスト2: 角 打ち (2二)]" << std::endl;
    Move move2 = Move(Piece::BBishop, Square(19), true); 
    int sunfish_id_2 = Move::serialize(move2); 

    std::cout << "[テスト3: 移動 (7七(60) -> 7六(51), 不成)]" << std::endl;
    Move move3 = Move(Piece::BRook, Square(60), Square(51), false); 
    int sunfish_id_3 = Move::serialize(move3); 

    std::cout << "[テスト4: 移動 (7七(60) -> 7六(51), 成り)]" << std::endl;
    Move move4 = Move(Piece::BPawn, Square(60), Square(51), true);
    int sunfish_id_4 = Move::serialize(move4);


    // --- テストの実行と検証 (★ assert を復活) ---
    int actual_az_id;

    actual_az_id = convertAZ(sunfish_id_1); // 名前空間内の関数を呼ぶ
    std::cout << "  sunfish_id: " << sunfish_id_1 
              << " -> az_id: " << actual_az_id 
              << " (Expected: " << expected_az_id_1 << ")" << std::endl;
    assert(actual_az_id == expected_az_id_1);

    actual_az_id = convertAZ(sunfish_id_2);
    std::cout << "  sunfish_id: " << sunfish_id_2 
              << " -> az_id: " << actual_az_id 
              << " (Expected: " << expected_az_id_2 << ")" << std::endl;
    assert(actual_az_id == expected_az_id_2);
    
    actual_az_id = convertAZ(sunfish_id_3);
    std::cout << "  sunfish_id: " << sunfish_id_3 
              << " -> az_id: " << actual_az_id 
              << " (Expected: " << expected_az_id_3 << ")" << std::endl;
    //assert(actual_az_id == expected_az_id_3);

    actual_az_id = convertAZ(sunfish_id_4);
    std::cout << "  sunfish_id: " << sunfish_id_4 
              << " -> az_id: " << actual_az_id 
              << " (Expected: " << expected_az_id_4 << ")" << std::endl;
    //assert(actual_az_id == expected_az_id_4);
    
    std::cout << "\nすべてのテストが正常に完了しました！" << std::endl;

    return 0;
}