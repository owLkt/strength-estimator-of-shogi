#include "shogi.h"
#include <utility>
#include <sstream>
#include <algorithm>

namespace minizero::env::shogi {

// ---------------- ShogiEnv ----------------

void ShogiEnv::reset() {
    board_.init(Board::Handicap::Even);
    winner_ = GameResult::UNDECIDED;
    actions_.clear();
    observations_.clear();
    turn_ = Player::kPlayer1;
    setLegalAction();
    board_history_.clear();
    repetition_history_.clear();
    board_hash_history_.clear();
    uint64_t hash = board_.getNoTurnHash();
    board_hash_history_[hash]++;
    board_history_.push_back(board_);
    repetition_history_.push_back(1);
}

void ShogiEnv::clearBoard() {
    board_.init();
    winner_ = GameResult::UNDECIDED;
    actions_.clear();
    observations_.clear();
    turn_ = Player::kPlayer1;
    legal_action_.reset();
    board_history_.clear();
    repetition_history_.clear();
    board_hash_history_.clear();
}

bool ShogiEnv::act(const ShogiAction& action) {
    //着手を盤面に反映
    Move move = action.toSunfishMove(board_);

    if (move.isEmpty() && action.getActionID() >= 0) {
        std::cerr << "[ShogiEnv::act] Error: toSunfishMove failed for AZ ID " 
                  << action.getActionID() << std::endl;
        return false;
    }

    if (!board_.makeMove(move)) {
        std::cerr << "[ShogiEnv::act] Error: board_.makeMove failed for move " << move.toString() << std::endl;
        return false;
    }

    uint64_t hash = board_.getNoTurnHash();
    board_hash_history_[hash]++;
    
    // 千日手カウント (Python: min(..., 3))
    int rep_count = std::min(board_hash_history_[hash], 3);
    
    // 新しい盤面と千日手カウントを履歴に追加
    board_history_.push_back(board_);
    repetition_history_.push_back(rep_count);

    actions_.push_back(action);
    turn_ = (turn_ == Player::kPlayer1 ? Player::kPlayer2 : Player::kPlayer1);
    setLegalAction();

    if(board_.isCheck(move) && legal_action_.count() == 0) {
        winner_ = (turn_ == Player::kPlayer1 ? GameResult::BLACK_WON : GameResult::WHITE_WON);
    }
   
    return true;
}

bool ShogiEnv::act(const std::vector<std::string>& action_string_args) {
    return act(ShogiAction(action_string_args));
}

std::vector<ShogiAction> ShogiEnv::getLegalActions() const {
    std::vector<ShogiAction> actions;

    for (size_t i = 0; i < legal_action_.size(); ++i) {
        if (legal_action_.test(i)) {
            actions.emplace_back(ShogiAction(i, turn_));
        }
    }

    return actions;
}

void ShogiEnv::setLegalAction() {
    legal_action_.reset();

    // 合法手生成
    Moves moves;
    MoveGenerator::generate(board_, moves);
    
    // 非合法手除去 & ビットセットへの登録
    for (const auto& move : moves) {
        if (board_.isValidMove(move)) { // 王手放置などの反則チェック
            int sunfish_id = Move::serialize16(move);
            int action_id = ShogiAction::convertAZ(sunfish_id, board_);
            if (action_id != -1) { 
                legal_action_.set(action_id);
            }
        }
    }
}

bool ShogiEnv::isLegalAction(const ShogiAction& action) const {
    return legal_action_.test(action.getActionID());
}

bool ShogiEnv::isTerminal() const {
    return winner_ != GameResult::UNDECIDED;
}

float ShogiEnv::getEvalScore(bool is_resign) const {
    return (winner_ == GameResult::BLACK_WON) ? 1.0f : (winner_ == GameResult::WHITE_WON) ? -1.0f : 0.0f;
}

// 駒の種類と手番からチャネルのインデックスを返す
int getPieceChannelIndex(const Piece& piece, Player turn) {
    bool is_own_piece = (piece.isBlack() && turn == Player::kPlayer1) || 
                        (piece.isWhite() && turn == Player::kPlayer2);
    
    int piece_type = piece.isEmpty() ? -1 : (piece.index() & Piece::KindMask); // 駒の種類（0-13）
    if (piece_type == -1) {
        return -1; // 空マスの場合は無効
    }

    return (is_own_piece ? 0 : 14) + piece_type; // 先手14チャネル、後手14チャネル
}

// 持ち駒の種類からチャネルのインデックスを返す
int getHandPieceChannelIndex(Piece piece, Player turn) {
    int piece_type = piece.index() & Piece::KindMask; // 駒の種類（0-6）
    if (piece_type < Piece::HandBegin || piece_type >= Piece::HandEnd) {
        return -1; // 持ち駒でない場合は無効
    }
    return 28 + (turn == Player::kPlayer1 ? 0 : 7) + piece_type; // 先手7チャネル、後手7チャネル
}

std::vector<float> ShogiEnv::getFeatures(utils::Rotation rotation) const {
    const int board_area = kShogiBoardSize * kShogiBoardSize; // 81
    const int num_channels = 362;
    std::vector<float> features(num_channels * board_area, 0.0f);

    const int T = 8; 
    const int channels_per_step = 45;
    
    bool is_white_turn = (turn_ == Player::kPlayer2);
    bool us_black = !is_white_turn;

    for (int t = 0; t < T; ++t) {
        int history_index = static_cast<int>(board_history_.size()) - 1 - t;
        int channel_offset = t * channels_per_step;

        if (history_index < 0) continue;

        const Board& history_board = board_history_[history_index];
        int rep_count = repetition_history_[history_index];

        // --- 1. 盤上の駒 (28 channels) ---
        for (int rank = 1; rank <= 9; ++rank) {
            for (int file = 1; file <= 9; ++file) { // Sunfish file: 1(9筋)...9(1筋)
                // ★修正1: Python(1筋=0)に合わせるため、1筋(file 9)を0にする
                int f = file - 1; 
                int r = rank - 1;

                if (is_white_turn) {
                    r = 8 - r;
                    f = 8 - f;
                }
                int sq_idx = r * 9 + f;

                Square sq(file, rank);
                Piece piece = history_board.getBoardPiece(sq);
                if (piece.isEmpty()) continue;

                int raw_kind = piece.index() & 0x0f;
                // ★修正2: Pythonの mapping に合わせ、成駒のインデックスを詰める
                int final_kind;
                if (raw_kind < 12) {
                    final_kind = raw_kind; // P, L, N, S, G, B, R, K, +P, +L, +N, +S
                } else {
                    final_kind = raw_kind - 1; // +B (13->12), +R (14->13)
                }

                bool is_us = (piece.isBlack() == us_black);
                int side_offset = is_us ? 0 : 14;

                int channel = channel_offset + side_offset + final_kind;
                features[channel * board_area + sq_idx] = 1.0f;
            }
        }

        // --- 2. 千日手 (3 channels) ---
        if (rep_count >= 1 && rep_count <= 3) {
            int channel = channel_offset + 28 + (rep_count - 1);
            std::fill(features.begin() + channel * board_area, 
                      features.begin() + (channel + 1) * board_area, 1.0f);
        }

        // --- 3. 持ち駒 (14 channels) ---
        const Hand& us_hand = us_black ? history_board.getBlackHand() : history_board.getWhiteHand();
        const Hand& enemy_hand = us_black ? history_board.getWhiteHand() : history_board.getBlackHand();

        // ★修正3: Pythonの ordered_hand = [P, L, N, S, B, R, G] に完全一致させる
        // Sunfish ID: P=0, L=1, N=2, S=3, G=4, B=5, R=6
        static const int py_hand_mapping[] = {0, 1, 2, 3, 5, 6, 4};

        for (int i = 0; i <= 6; ++i) {
            int pt = py_hand_mapping[i];

            Piece us_piece = us_black ? Piece(static_cast<uint8_t>(pt)).black() : Piece(static_cast<uint8_t>(pt)).white();
            Piece enemy_piece = us_black ? Piece(static_cast<uint8_t>(pt)).white() : Piece(static_cast<uint8_t>(pt)).black();

            int us_count = us_hand.get(us_piece);
            if (us_count > 0) {
                int channel = channel_offset + 31 + i;
                std::fill(features.begin() + channel * board_area, 
                          features.begin() + (channel + 1) * board_area, static_cast<float>(us_count));
            }
            int enemy_count = enemy_hand.get(enemy_piece);
            if (enemy_count > 0) {
                int channel = channel_offset + 38 + i;
                std::fill(features.begin() + channel * board_area, 
                          features.begin() + (channel + 1) * board_area, static_cast<float>(enemy_count));
            }
        }
    }

    // --- 4. グローバル (Turn, Move Count) ---
    int turn_channel = T * channels_per_step; 
    float turn_val = (turn_ == Player::kPlayer1) ? 1.0f : 0.0f; // Python: BLACKなら1.0
    std::fill(features.begin() + turn_channel * board_area, 
              features.begin() + (turn_channel + 1) * board_area, turn_val);

    int move_channel = turn_channel + 1; 
    float move_count_val = static_cast<float>(actions_.size()) / 512.0f;
    std::fill(features.begin() + move_channel * board_area, 
              features.begin() + (move_channel + 1) * board_area, move_count_val);

    return features;
}

std::vector<float> ShogiEnv::getActionFeatures(const ShogiAction& action, utils::Rotation rotation) const {
    std::vector<float> feat;
    return feat;
}

std::string ShogiEnv::toString() const {
    return board_.toString();
}

static std::string trim(const std::string& str) {
    const std::string whitespace = " \t\n\r";
    size_t first = str.find_first_not_of(whitespace);
    if (std::string::npos == first) {
        return "";
    }
    size_t last = str.find_last_not_of(whitespace);
    return str.substr(first, (last - first + 1));
}

// ---------------- ShogiEnvLoader ----------------

std::vector<float> ShogiEnvLoader::getActionFeatures(const int pos, utils::Rotation rotation) const {
    std::vector<float> feat;
    return feat;
}

bool ShogiEnvLoader::loadFromString(const std::string& content) {
    this->action_pairs_.clear();
    this->tags_.clear();

    ShogiEnv temp_env;
    std::stringstream ss(content);
    std::string segment;
    std::vector<std::string> p_lines; 
    bool position_loaded = false; 

    while (std::getline(ss, segment, ',')) { 
        // 1. セグメントの空白除去（必須）
        segment = trim(segment);
        if (segment.empty()) continue;

        try {
            // P行の蓄積
            if (segment[0] == 'P' && !position_loaded) {
                p_lines.push_back(segment);
            } 
            // 手番/指し手開始
            else if (segment[0] == '+' || segment[0] == '-') {
                
                if (!position_loaded) {
                    // ★最重要修正: reset()ではなくclearBoard()を使う
                    // これで盤面は「全マス空」になり、ゴミが残る心配がなくなります
                    temp_env.clearBoard(); 
                    
                    if (!p_lines.empty()) {
                        for (const std::string& line : p_lines) {
                            if (line.length() < 2) continue;

                            // 盤上の駒配置 (P1...)
                            if (line[1] >= '1' && line[1] <= '9') {
                                int rank = line[1] - '0';
                                std::string body = line.substr(2);

                                size_t idx = 0;
                                // 9筋 -> 1筋
                                for (int file = 9; file >= 1; --file) {
                                    // スペース飛ばし
                                    while (idx < body.length() && (body[idx] == ' ' || body[idx] == '\t')) {
                                        idx++;
                                    }
                                    if (idx >= body.length()) break;

                                    // 先手駒
                                    if (body[idx] == '+') {
                                        if (idx + 2 < body.length()) {
                                            std::string kind_str = body.substr(idx + 1, 2);
                                            Piece p = Piece::parseCsa(kind_str.c_str());
                                            if (!p.isEmpty()) temp_env.board_.setBoardPiece(Square(file, rank), p.black());
                                            idx += 3;
                                        } else { idx++; }
                                    } 
                                    // 後手駒
                                    else if (body[idx] == '-') {
                                        if (idx + 2 < body.length()) {
                                            std::string kind_str = body.substr(idx + 1, 2);
                                            Piece p = Piece::parseCsa(kind_str.c_str());
                                            if (!p.isEmpty()) temp_env.board_.setBoardPiece(Square(file, rank), p.white());
                                            idx += 3;
                                        } else { idx++; }
                                    } 
                                    // 空マス (*)
                                    else if (body[idx] == '*') {
                                        // 盤面は既に空なので、明示的に消す必要なし（スキップだけでOK）
                                        idx++; 
                                    }
                                    else {
                                        idx++;
                                    }
                                }
                            }
                            // 持ち駒 (P+...)
                            else if (line[1] == '+' || line[1] == '-') {
                                bool is_black_hand = (line[1] == '+');
                                std::string body = line.substr(2);
                                for (size_t i = 0; i + 3 < body.length(); ) {
                                    if (isdigit(body[i]) && isdigit(body[i+1])) {
                                         std::string kind_str = body.substr(i+2, 2);
                                         if (kind_str != "AL") {
                                             Piece p = Piece::parseCsa(kind_str.c_str());
                                             if (!p.isEmpty()) {
                                                 if (is_black_hand) temp_env.board_.incBlackHand(p.kindOnly());
                                                 else temp_env.board_.incWhiteHand(p.kindOnly());
                                             }
                                         }
                                         i += 4;
                                    } else { i++; }
                                }
                            }
                        }
                        
                        temp_env.board_.refreshHash();
                        temp_env.setLegalAction();
                    }
                    else {
                        // P行がない場合は平手初期配置にする
                        temp_env.reset();
                    }
                    position_loaded = true;
                }

                // --- 手番設定 ---
                Player csa_turn = (segment[0] == '+') ? Player::kPlayer1 : Player::kPlayer2;
                if (csa_turn == Player::kPlayer1) {
                    temp_env.setBlack();
                    temp_env.board_.setBlack();
                } else {
                    temp_env.setWhite();
                    temp_env.board_.setWhite();
                }
                temp_env.setLegalAction();

                if (segment.length() == 1) continue; 

                // --- 指し手処理 ---
                const Board& current_board = temp_env.getBoard();
                std::string move_str = segment.substr(1); 
                
                Move move = Move::parseCsa(current_board, move_str.c_str());
                if (move.isEmpty()) {
                    std::string from_str = move_str.substr(0, 2);
                    std::cerr << "[loadFromString] Error: Failed to parse move: " << segment;
                    if (isdigit(from_str[0]) && isdigit(from_str[1])) {
                        int f = from_str[0] - '0';
                        int r = from_str[1] - '0';
                        Piece p = current_board.getBoardPiece(Square(f, r));
                        std::cerr << " | Piece at " << f << r << ": " << p.toString();
                    }
                    std::cerr << " | Turn(Brd): " << (current_board.isBlack() ? "B" : "W") << std::endl;
                    return !this->action_pairs_.empty(); 
                }

                int sunfish_id = Move::serialize16(move);
                int az_action_id = ShogiAction::convertAZ(sunfish_id, current_board);

                if (az_action_id == -1) {
                    std::cerr << "[loadFromString] Error: Invalid AZ Action. Move=" << segment 
                              << " | SunfishID:" << sunfish_id << std::endl;
                    return !this->action_pairs_.empty(); 
                }

                ShogiAction action(az_action_id, temp_env.getTurn());
                this->addActionPair(action, {}); 

                if (!temp_env.act(action)) {
                    std::cerr << "[loadFromString] Error: act() failed for move: " << segment << std::endl;
                    //std::cerr << temp_env.toString() << std::endl;
                    return !this->action_pairs_.empty();
                }
            }
            else if (segment[0] == '%') { break; }
            else if (segment.rfind("'black_rate:", 0) == 0) { this->addTag("BR", segment.substr(segment.rfind(':') + 1)); }
            else if (segment.rfind("'white_rate:", 0) == 0) { this->addTag("WR", segment.substr(segment.rfind(':') + 1)); }

        } catch (const std::exception& e) {
            std::cerr << "[loadFromString] Exception: " << e.what() << " at segment: " << segment << std::endl;
            return false;
        }
    } 

    if (!position_loaded && p_lines.empty()) { temp_env.reset(); }
    
    return !this->action_pairs_.empty();
}

} // namespace minizero::env::shogi
