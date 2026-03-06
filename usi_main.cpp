#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <memory>
#include <algorithm>
#include <map>
#include <fstream>
#include <cmath>
#include <limits>
#include <iomanip>
#include <numeric>
#include <filesystem>

// Torch & Minizero headers
#include <torch/torch.h>
#include "strength/misc/st_configuration.h"
#include "minizero/environment/shogi/shogi.h"
#include "minizero/minizero/config/configuration.h"
#include "strength/misc/st_actor.h"
#include "strength/misc/strength_network.h" 

using namespace minizero;
using namespace minizero::env::shogi;
using namespace minizero::env;

class MiniZeroUSIEngine {
    std::shared_ptr<strength::StrengthNetwork> network;
    std::shared_ptr<strength::StActor> actor;
    ShogiEnv env; 
    
    std::vector<std::string> current_game_history;
    
    // --- 適応型強さ調整用の変数 ---
    const std::string memory_file = "player_strength_memory.txt";
    double total_opponent_score = 0.0; 
    long long total_opponent_moves = 0;     
    
    float current_target_score = 0.0f; 

    int manual_rank = -1; // -1 の時はアダプティブ、0〜7 の時は固定ランク
    
    // AIが担当した手番 (None:未定, Player1:先手, Player2:後手)
    Player engine_color = Player::kPlayerNone;

    // ★修正: 最新の学習結果を反映
    const std::vector<float> rank_benchmarks = {
        1.1150f,  // Rank 0 (Elo 1500)
        3.0156f,  // Rank 1
        3.7231f,  // Rank 2
        5.3674f,  // Rank 3
        6.6582f,  // Rank 4
        7.7593f,  // Rank 5
        10.1883f,  // Rank 6
        12.7975f   // Rank 7 (Elo 4300)
    };

public:
    void run() {
        std::string line, cmd;
        while (std::getline(std::cin, line)) {
            std::stringstream ss(line);
            if (!(ss >> cmd)) continue;

            if (cmd == "usi") {
                std::cout << "id name MiniZero_Adaptive_Shogi" << std::endl;
                std::cout << "id author YourName" << std::endl;
                std::cout << "option name Rank type spin default -1 min -1 max 7" << std::endl;
                std::cout << "usiok" << std::endl;
            } 
            else if (cmd == "setoption") {
                // ★GUIからの設定を受け取る
                std::string name_label, name, value_label;
                int value;
                ss >> name_label >> name >> value_label >> value;
                if (name == "Rank") {
                    manual_rank = value;
                    std::cerr << "info string Manual Rank set to: " << manual_rank << std::endl;
                }
            }
            else if (cmd == "isready") {
                if (loadModel()) {
                    loadMemory(); 
                    std::cout << "readyok" << std::endl; 
                }
            } 
            else if (cmd == "usinewgame") {
                env.reset();
                if (actor) actor->reset();
                current_game_history.clear();
                engine_color = Player::kPlayerNone; // 手番リセット
                strength::actor_select_action_by_bt = true;  // BT調整をオンにする
                
                // --- 強さの設定 ---
                if (manual_rank != -1) {
                    const float offset = 6.0f;
                    // 手動指定がある場合（固定ランク対局）
                    current_target_score = rank_benchmarks[manual_rank] - offset;
                    std::cout << "info string [Fixed Mode] Rank: " << manual_rank << std::endl;
                } else if (total_opponent_moves == 0) {
                    // 1局目: 相手のデータがないので、rank7に設定
                    current_target_score = rank_benchmarks.back(); 
                    std::cout << "info string [Adaptive] First game. Using Max rank." << std::endl;
                } else {
                    // 2局目以降: 蓄積された相手の平均スコアに基づいて手加減する
                    double avg_score = total_opponent_score / total_opponent_moves;
                    int estimated_rank = findClosestRank(avg_score);
                    current_target_score = rank_benchmarks[estimated_rank];
                    
                    std::cout << "info string [Adaptive] Opponent Avg: " << avg_score << " -> Set Rank: " << estimated_rank << std::endl;
                }
                //current_target_score = rank_benchmarks.back();
                strength::cand_strength.clear(); 
                strength::cand_strength.resize(512, current_target_score);
                //std::cout << " -> Set Rank: " << current_target_score << std::endl;
            } 
            else if (cmd == "position") {
                handlePosition(ss); 
            } 
            else if (cmd == "go") {
                handleGo(); 
            } 
            else if (cmd == "gameover") {
                handleGameOver(ss);
            }
            else if (cmd == "quit") {
                break;
            }
            else if (cmd == "reset_memory") {
                resetMemory();
            }
        }
    }

private:
    int findClosestRank(float score) {
        int best_idx = 0;
        float min_diff = std::numeric_limits<float>::max();
        for (size_t i = 0; i < rank_benchmarks.size(); ++i) {
            float diff = std::abs(score - rank_benchmarks[i]);
            if (diff < min_diff) {
                min_diff = diff;
                best_idx = i;
            }
        }
        return best_idx;
    }

    void handleGo() {
        if (!actor) return;

        engine_color = env.getTurn();

        auto legal_actions = env.getLegalActions();
        if (legal_actions.empty()) { std::cout << "bestmove resign" << std::endl; return; }

        Action action = actor->think(true, false); 
        bool is_legal = false;
        for (const auto& a : legal_actions) {
            if (a.getActionID() == action.getActionID()) { is_legal = true; break; }
        }
        if (!is_legal) action = legal_actions[0]; 

        if (action.getActionID() != -1) {
            ShogiAction best_action = static_cast<const ShogiAction&>(action);
            Move m = best_action.toSunfishMove(env.getBoard());
            std::cout << "bestmove " << moveToUsi(m) << std::endl;
        } else {
            std::cout << "bestmove resign" << std::endl;
        }
    }

    void handlePosition(std::stringstream& ss) {
        std::string type, move_str;
        ss >> type; 
        env.reset();
        actor->reset();
        current_game_history.clear();
        std::vector<std::string> move_str_list;
        if (ss >> move_str && move_str == "moves") {
            while (ss >> move_str) { move_str_list.push_back(move_str); }
        }
        for (const auto& m_str : move_str_list) {
            current_game_history.push_back(m_str);
            Move m = usiToMove(env.getBoard(), m_str);
            if (m.isEmpty()) continue;
            int az_id = ShogiAction::convertAZ(Move::serialize16(m), env.getBoard());
            if (az_id != -1) {
                ShogiAction action(az_id, env.getTurn());
                env.act(action);
                actor->act(action);
            }
        }
    }

    void handleGameOver(std::stringstream& ss) {
        if (engine_color == Player::kPlayerNone) return;

        std::string result;
        ss >> result; // lose, win, draw など

        std::cerr << "info string Analyzing game with Value-Weighting..." << std::endl;

        ShogiEnv replay_env;
        replay_env.reset();
        
        std::vector<float> opponent_scores;

        // 棋譜を再現して評価値を再計算
        for (const auto& move_usi : current_game_history) {
            auto features = replay_env.getFeatures();
            network->pushBack(features);
            auto outputs = network->forward();
            auto s_output = std::static_pointer_cast<strength::StrengthNetworkOutput>(outputs[0]);
            
            // 指し手の質の計算（Value補正込み）
            float win_rate = (s_output->value_ + 1.0f) / 2.0f; 
            const float offset = 6.0f;
            float raw_score = s_output->score_;
            float adjusted_score = (raw_score + offset) + (win_rate * 2.5f);

            // 相手の手番の時のスコアのみを収集
            if (replay_env.getTurn() != engine_color) {
                opponent_scores.push_back(adjusted_score);
            }

            Move m = usiToMove(replay_env.getBoard(), move_usi);
            int az_id = ShogiAction::convertAZ(Move::serialize16(m), replay_env.getBoard());
            if (az_id != -1) replay_env.act(ShogiAction(az_id, replay_env.getTurn()));
        }

        // 今回の対局データを累積変数に加算
        if (!opponent_scores.empty()) {
            // 手動設定（manual_rank != -1）の時は、統計データそのものを更新しないようにする
            if (manual_rank == -1) {
                float game_sum = std::accumulate(opponent_scores.begin(), opponent_scores.end(), 0.0f);
                total_opponent_score += game_sum;
                total_opponent_moves += opponent_scores.size();

                saveMemory(); // 保存
            
                double total_avg = total_opponent_score / total_opponent_moves;
                std::cout << "info string [Adaptive] Stats Updated. Global Avg: " << total_avg 
                          << " (Total Moves: " << total_opponent_moves << ")" << std::endl;
            } else {
                std::cout << "info string [Fixed Mode] Game finished. Adaptive stats not updated." << std::endl;
            }
        }
    }

    void saveMemory() {
        std::ofstream ofs(memory_file);
        if (ofs) {
            ofs << total_opponent_score << " " << total_opponent_moves << std::endl;
        }
    }

    void loadMemory() {
        if (std::filesystem::exists(memory_file)) {
            std::ifstream ifs(memory_file);
            if (ifs >> total_opponent_score >> total_opponent_moves) {
                std::cerr << "info string Loaded memory. Total moves: " << total_opponent_moves << std::endl;
            }
        }
    }

    void resetMemory() {
        if (std::filesystem::exists(memory_file)) std::filesystem::remove(memory_file);
        total_opponent_score = 0;
        total_opponent_moves = 0;
        std::cerr << "info string Memory reset." << std::endl;
    }

    Move usiToMove(const Board& board, const std::string& usi) {
        if (usi.size() < 4) return Move::empty();
        auto get_f = [](char c) { return '9' - c; };
        auto get_r = [](char c) { return c - 'a'; };
        if (usi[1] == '*') {
            static std::map<char, uint8_t> p_map = {{'P',0},{'L',1},{'N',2},{'S',3},{'G',4},{'B',5},{'R',6}};
            int f = get_f(usi[2]); int r = get_r(usi[3]);
            Square to_sq(f * 9 + r);
            Player p = board.isBlack() ? Player::kPlayer1 : Player::kPlayer2;
            uint8_t piece_index = p_map[usi[0]] + (p == Player::kPlayer2 ? 16 : 0);
            return Move(Piece(piece_index), to_sq, true);
        }
        int f1 = get_f(usi[0]); int r1 = get_r(usi[1]);
        int f2 = get_f(usi[2]); int r2 = get_r(usi[3]);
        Square from_sq(f1 * 9 + r1);
        Square to_sq(f2 * 9 + r2);
        return Move(board.getBoardPiece(from_sq), from_sq, to_sq, (usi.size() > 4 && usi[4] == '+'), false);
    }
    std::string moveToUsi(const Move& m) {
        if (m.isEmpty()) return "resign";
        auto to_s = [](int idx) { int f = idx/9; int r = idx%9; return std::string(1, '9'-f) + (char)('a'+r); };
        std::string res = "";
        if (m.isHand()) {
            static const char p_names[] = "PLNSGBR";
            res += p_names[m.piece().index() & 0x07]; res += "*";
        } else { res += to_s(m.from().index()); }
        res += to_s(m.to().index()); if (m.promote()) res += "+";
        return res;
    }
    bool loadModel() {
        if (actor) return true;
        try {
            network = std::make_shared<strength::StrengthNetwork>();
            network->loadModel(config::nn_file_name, 0); 
            actor = std::make_shared<strength::StActor>(1000000);
            actor->setNetwork(network);
            return true;
        } catch (const std::exception& e) { return false; }
    }
};

int main(int argc, char* argv[]) {
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    config::ConfigureLoader cl;
    config::setConfiguration(cl);    
    strength::setConfiguration(cl); 
    for (int i = 1; i < argc; ++i) { if (std::string(argv[i]) == "-conf" && i + 1 < argc) cl.loadFromFile(argv[i + 1]); }
    config::learner_batch_size = 1;      
    config::zero_num_parallel_games = 1; 
    MiniZeroUSIEngine engine;
    engine.run();
    return 0;
}