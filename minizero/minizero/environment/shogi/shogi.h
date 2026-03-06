#pragma once

#include "../base/base_env.h"
#include <map>
#include <string>
#include <vector>
#include "board.h"      
#include "move.h" 
#include "moves.h"
#include "MoveGenerator.h" 
#include "Piece.h"
#include "Square.h"
#include "Hand.h"
#include <bitset>
#include <iostream>

namespace minizero::env::shogi {

const std::string kShogiName = "shogi";
const int kShogiNumPlayer = 2;
const int kShogiBoardSize = 9;
const int kShogiBoardArea = kShogiBoardSize * kShogiBoardSize;
const int kShogiPolicySize = 11259;

// --- Action ---
class ShogiAction : public BaseAction {
public:
    ShogiAction() : BaseAction() {}
    ShogiAction(int action_id, Player player) : BaseAction(action_id, player) {}
    ShogiAction(const std::vector<std::string>& action_string_args) {
        assert(action_string_args.size() == 2);
        assert(action_string_args[0].size() == 1);
        player_ = charToPlayer(action_string_args[0][0]);
        assert(static_cast<int>(player_) > 0 && static_cast<int>(player_) <= kShogiNumPlayer);
    }

    inline Player nextPlayer() const override { return getNextPlayer(player_, kShogiNumPlayer); }
    inline std::string toConsoleString() const override {
        return "";
    }
    Move toSunfishMove(const Board& board) const { 
        if(action_id_ < 0)
            return Move::empty();
        uint32_t sunfish_id = convertSunfish(action_id_, board);
        return Move::deserialize16(sunfish_id, board);
    }

    static int fromSunfishMove(const Piece& piece, const Square& from, const Square& to, bool promote, bool safe = true) {
        Move move(piece, from, to, promote, safe);
        return Move::serialize16(move);
    }

    static int fromSunfishMove(const Piece& piece, const Square& to, bool safe = true) {
        Move move(piece, to, safe);
        return Move::serialize16(move);
    }

    static int map_dx_dy_to_direction_id(int dx, int dy) {
        // dy < 0 は「上」方向、dx < 0 は「左」方向。

        // 桂馬の動き (標準的な定義に戻す)
        if (dx == -1 && dy == -2) return 0; // 桂馬 (左)
        if (dx == 1 && dy == -2) return 1;  // 桂馬 (右)

        // 直線・斜めの動き
        int distance = 0;

        if (dx == 0 && dy < 0) { // 上
            distance = -dy; 
            return 2 + (distance - 1); 
        }
        if (dx == 0 && dy > 0) { // 下
            distance = dy;
            return 10 + (distance - 1); 
        }
        if (dy == 0 && dx < 0) { // 左
            distance = -dx;
            return 18 + (distance - 1); 
        }
        if (dy == 0 && dx > 0) { // 右
            distance = dx;
            return 26 + (distance - 1); 
        }

        if (dx < 0 && dy < 0 && dx == dy) { // 左上
            distance = -dx;
            return 34 + (distance - 1); 
        }
        if (dx > 0 && dy < 0 && dx == -dy) { // 右上
            distance = dx;
            return 42 + (distance - 1); 
        }
        if (dx < 0 && dy > 0 && dx == -dy) { // 左下
            distance = -dx;
            return 50 + (distance - 1); 
        }
        if (dx > 0 && dy > 0 && dx == dy) { // 右下
            distance = dx;
            return 58 + (distance - 1); 
        }

        return -1; 
    }

    /**
     * Rank-Major (Rank*9 + File) の座標系を受け取り、方向IDを計算する
     */
    static int get_direction_id(int from_sq, int to_sq) {
        // Rank-Major 前提: sq / 9 = Rank, sq % 9 = File
        int from_rank = from_sq / 9;
        int from_file = from_sq % 9;
        int to_rank = to_sq / 9;
        int to_file = to_sq % 9;

        int dx = to_file - from_file; 
        int dy = to_rank - from_rank; 

        return map_dx_dy_to_direction_id(dx, dy);
    }

    static int convertAZ(int sunfish_id, const Board& board) {
        Move move = Move::deserialize16(sunfish_id, board);
        bool is_white_move = board.isWhite();

        int f_idx = 8 - (move.from().index() / 9); 
        int r_idx = move.from().index() % 9;
        
        int tf_idx = 8 - (move.to().index() / 9);
        int rt_idx = move.to().index() % 9;

        int az_from = r_idx * 9 + f_idx; // これで 7g -> index 60 (Python一致)
        int az_to = rt_idx * 9 + tf_idx;

        int rotated_from = is_white_move ? (80 - az_from) : az_from;
        int rotated_to = is_white_move ? (80 - az_to) : az_to;

        if (move.isHand()) {
            // 駒打ち: (駒ID * 81) + rotated_to
            // Python piece_map: P:0, L:1, N:2, S:3, B:4, R:5, G:6
            static const int piece_map[] = {0, 1, 2, 3, 6, 4, 5}; // Sunfish ID -> Py ID
            int py_piece_type = piece_map[move.piece().index() & Piece::KindMask];
            return py_piece_type * 81 + rotated_to;
        } else {
            // 移動: (7 * 81) + (from * 132) + (move_id * 2) + promote
            int rf_r = rotated_from / 9; int rf_f = rotated_from % 9;
            int rt_r = rotated_to / 9;   int rt_f = rotated_to % 9;
            int dx = rt_f - rf_f;
            int dy = rt_r - rf_r;

            int move_id = map_dx_dy_to_direction_id(dx, dy);
            if (move_id == -1) return -1;

            int promote_offset = move.promote() ? 1 : 0;
            return (7 * 81) + (rotated_from * 132) + (move_id * 2) + promote_offset;
        }
    }

    /**
     * (逆変換用) Rank-Major座標系で移動先を計算
     */
    static int get_to_sq_from_direction(int from_sq, int direction_id) {
        int from_rank = from_sq / 9; 
        int from_file = from_sq % 9; 

        int dx = 0; int dy = 0; int distance = 0;

        if (direction_id == 0) { dx = -1; dy = -2; }      // 桂馬 (左)
        else if (direction_id == 1) { dx = 1; dy = -2; }  // 桂馬 (右)
        else if (direction_id >= 2 && direction_id <= 9) { distance = (direction_id - 2) + 1; dx = 0; dy = -distance; }
        else if (direction_id >= 10 && direction_id <= 17) { distance = (direction_id - 10) + 1; dx = 0; dy = distance; }
        else if (direction_id >= 18 && direction_id <= 25) { distance = (direction_id - 18) + 1; dx = -distance; dy = 0; }
        else if (direction_id >= 26 && direction_id <= 33) { distance = (direction_id - 26) + 1; dx = distance; dy = 0; }
        else if (direction_id >= 34 && direction_id <= 41) { distance = (direction_id - 34) + 1; dx = -distance; dy = -distance; }
        else if (direction_id >= 42 && direction_id <= 49) { distance = (direction_id - 42) + 1; dx = distance; dy = -distance; }
        else if (direction_id >= 50 && direction_id <= 57) { distance = (direction_id - 50) + 1; dx = -distance; dy = distance; }
        else if (direction_id >= 58 && direction_id <= 65) { distance = (direction_id - 58) + 1; dx = distance; dy = distance; }
        else { return -1; }

        int to_file = from_file + dx;
        int to_rank = from_rank + dy;

        if (to_file < 0 || to_file > 8 || to_rank < 0 || to_rank > 8) {
            // std::cerr << "Error: Move is off-board." << std::endl;
            return -1;
        }

        return to_rank * 9 + to_file; // Rank-Major
    }

    // AlphaZero -> Sunfish 変換
    static int convertSunfish(int az_action_id, const Board& pos) {
        bool is_white_move = pos.isWhite();
        // AIの座標 (Rank-Major) を Sunfishの座標 (File-Major) に戻す関数
        auto to_sunfish_sq = [](int az_sq) {
            int r = az_sq / 9; // 段 (0..8)
            int f = az_sq % 9; // 筋 (0..8)
            return (8 - f) * 9 + r;
        };

        if (az_action_id < 7 * kShogiBoardArea) {
            int py_piece_type = az_action_id / kShogiBoardArea;
            int rotated_to = az_action_id % 81;

            // 相対から絶対へ復元
            int az_to = is_white_move ? (80 - rotated_to) : rotated_to;

            // Python piece_map の逆変換
            static const int rev_piece_map[] = {0, 1, 2, 3, 5, 6, 4}; // Py -> Sunfish
            Piece piece(static_cast<uint8_t>(rev_piece_map[py_piece_type])); 
            
            return fromSunfishMove(piece, Square(to_sunfish_sq(az_to)));
        
        } else {
            int move_id_raw = az_action_id - (7 * kShogiBoardArea);
            int rotated_from = move_id_raw / 132;
            int move_type_id = move_id_raw % 132;
            int direction_id = move_type_id / 2;
            bool promote = (move_type_id % 2) == 1;
        
            int rotated_to = get_to_sq_from_direction(rotated_from, direction_id);
            if (rotated_to == -1) return 0;

            // 相対から絶対へ復元
            int az_from = is_white_move ? (80 - rotated_from) : rotated_from;
            int az_to = is_white_move ? (80 - rotated_to) : rotated_to;

            int sunfish_from = to_sunfish_sq(az_from);
            int sunfish_to = to_sunfish_sq(az_to);

            Piece piece_on_board = pos.getBoardPiece(Square(sunfish_from));
            return fromSunfishMove(piece_on_board.unpromote(), Square(sunfish_from), Square(sunfish_to), promote);
        }
    };
};

// --- Env ---
class ShogiEnv : public BaseBoardEnv<ShogiAction> {
public:
    ShogiEnv() : BaseBoardEnv<ShogiAction>(kShogiBoardSize) { reset(); }

    void reset() override;
    void clearBoard();
    bool act(const ShogiAction& action) override;
    bool act(const std::vector<std::string>& action_string_args) override;
    std::vector<ShogiAction> getLegalActions() const override;
    void setLegalAction();
    bool isLegalAction(const ShogiAction& action) const override;
    bool isTerminal() const override;
    float getReward() const override {return 0.0f;}
    float getEvalScore(bool is_resign = false) const override;
    std::vector<float> getFeatures(utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::vector<float> getActionFeatures(const ShogiAction& action, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    std::string toString() const override;
    Board board_; // 現在の局面
    const Board& getBoard() const { return board_; }
    inline bool isBlack() const { return board_.isBlack(); }
    inline bool isWhite() const { return board_.isWhite(); }
    inline void setBlack() { board_.setBlack(); }
    inline void setWhite() { board_.setWhite(); }

    inline std::string name() const override { return kShogiName; }
    inline int getNumPlayer() const override { return kShogiNumPlayer; }
    inline int getRotatePosition(int position, utils::Rotation rotation) const override { return position; }
    inline int getRotateAction(int action_id, utils::Rotation rotation) const override { return action_id; }

    // AlphaZero 互換のチャネル数
    inline int getNumInputChannels() const override { return 362; } // 将棋用
    inline int getPolicySize() const override { return /* 全行動数 */ kShogiPolicySize; }
    inline int getNumActionFeatureChannels() const override { return 0; }
    void setTurn(Player p) { turn_ = p; }

private:
    GameResult winner_ = GameResult::UNDECIDED;
    std::bitset</*policy size*/ kShogiPolicySize> legal_action_;
    std::vector<Board> board_history_;
    std::vector<int> repetition_history_;
    std::map<uint64_t, int> board_hash_history_;
};

// --- EnvLoader ---
class ShogiEnvLoader : public BaseBoardEnvLoader<ShogiAction, ShogiEnv> {
public:
    std::vector<float> getActionFeatures(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    inline std::vector<float> getValue(const int pos) const { return {getReturn()}; } //
    inline std::string name() const override { return kShogiName; }
    bool loadFromString(const std::string& content) override; 
    inline int getPolicySize() const override { return kShogiPolicySize ; }
    inline int getRotatePosition(int position, utils::Rotation rotation) const override { return position; }
    inline int getRotateAction(int action_id, utils::Rotation rotation) const override { return action_id; }
};

} // namespace minizero::env::shogi
