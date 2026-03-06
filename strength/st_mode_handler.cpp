#include "st_mode_handler.h"
#include "evaluator.h"
#include "game_wrapper.h"
#include "git_info.h"
#include "st_actor.h"
#include "st_actor_group.h"
#include "st_configuration.h"
#include "st_console.h"
#include "time_system.h"
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "cnpy.h"

namespace strength {

using namespace minizero;
using namespace minizero::utils;

inline int getRankIndexFromDir(const std::string& path_str) {
    try {
        std::filesystem::path path(path_str);
        std::string dirname = path.filename().string(); 
        
        size_t split_pos = dirname.find('_');
        if (split_pos == std::string::npos) return -1;
        
        int elo = std::stoi(dirname.substr(0, split_pos)); 
        
        const int min_elo = strength::nn_rank_min_elo;
        const int interval = strength::nn_rank_elo_interval;

        if (elo < min_elo) return -1;
        
        return (elo - min_elo) / interval;
    } catch (...) {
        return -1;
    }
}

StModeHandler::StModeHandler()
{
    RegisterFunction("evaluator", this, &StModeHandler::runEvaluator);
    RegisterFunction("mcts_acc", this, &StModeHandler::runMCTSAccuracy);
}
void StModeHandler::loadNetwork(const std::string& nn_file_name, int gpu_id /* = 0 */)
{
    network_ = std::make_shared<StrengthNetwork>();
    network_->loadModel(nn_file_name, gpu_id);
}
void StModeHandler::runConsole()
{
    StConsole console;
    std::string command;
    console.initialize();
    std::cerr << "Successfully started console mode" << std::endl;
    while (getline(std::cin, command)) {
        if (command == "quit") { break; }
        console.executeCommand(command);
    }
}

void StModeHandler::runSelfPlay()
{
    STActorGroup ag;
    ag.run();
}

void StModeHandler::runZeroTrainingName()
{
    std::cout << Environment().name()                  // name for environment
              << "_" << getNetworkAbbeviation()        // network & training algorithm
              << "_" << config::nn_num_blocks << "b"   // number of blocks
              << "x" << config::nn_num_hidden_channels // number of hidden channels
              << "-" << GIT_SHORT_HASH << std::endl;   // git hash info
}

void StModeHandler::runEvaluator()
{
    Evaluator evaluator;
    evaluator.run();
}
void StModeHandler::runMCTSAccuracy()
{
    if (actor_select_action_by_bt) {
        loadNetwork(config::nn_file_name);
        
        std::string cand_dir = strength::candidate_sgf_dir;
        std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") 
                  << "Loading candidate data from .npy directories: " << cand_dir << std::endl;

        std::map<int, std::vector<std::pair<float, float>>> candidate_results;

if (std::filesystem::exists(cand_dir)) {
            // 指定されたパスそのものからランクを取得
            int rank_idx = getRankIndexFromDir(cand_dir);
            
            if (rank_idx >= 0) {
                // そのフォルダ直下の features.npy を探す
                std::string f_path = cand_dir + "/features.npy";
                
                if (std::filesystem::exists(f_path)) {
                    try {
                        cnpy::NpyArray features_arr = cnpy::npy_load(f_path);
                        float* features_ptr = features_arr.data<float>();
                        size_t num_positions = features_arr.shape[0];

                        if (num_positions > 0) {
                            // 特徴量次元計算
                            size_t feature_dim = 1;
                            for(size_t i=1; i<features_arr.shape.size(); ++i) feature_dim *= features_arr.shape[i];

                            std::shared_ptr<StrengthNetwork> network = std::static_pointer_cast<StrengthNetwork>(network_);
                            
                            // メモリ対策：batch_size を config から取得、または安全な値(128等)に固定
                            size_t batch_size = (config::learner_batch_size > 0) ? config::learner_batch_size : 128;
                            
                            for (size_t start_idx = 0; start_idx < num_positions; start_idx += batch_size) {
                                size_t end_idx = std::min(start_idx + batch_size, num_positions);
                                for (size_t i = start_idx; i < end_idx; ++i) {
                                    std::vector<float> feat(features_ptr + i * feature_dim, 
                                                            features_ptr + (i + 1) * feature_dim);
                                    network->pushBack(feat);
                                }
                                auto batch_out = network->forward();
                                
                                // 結果を集計
                                for (auto& out : batch_out) {
                                    auto s_out = std::static_pointer_cast<StrengthNetworkOutput>(out);
                                    float score = s_out->score_;
                                    float weight = strength::bt_use_weight ? s_out->weight_ : 1.0f;
                                    candidate_results[rank_idx].push_back({score * weight, weight});
                                }
                            }
                            std::cerr << "Loaded " << num_positions << " positions from rank " << rank_idx << std::endl;
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "[Error] Failed to process " << cand_dir << ": " << e.what() << std::endl;
                    }
                } else {
                    std::cerr << "[Error] features.npy not found in " << cand_dir << std::endl;
                }
            } else {
                std::cerr << "[Error] Could not determine rank index from path: " << cand_dir << std::endl;
            }
        } else {
            std::cerr << "[Error] Candidate directory not found: " << cand_dir << std::endl;
        }

        // 強さ (cand_strength) の更新
        std::cerr << "Candidate Strengths: ";
        for (auto& kv : candidate_results) {
            int rank_idx = kv.first;
            double sum_weighted_score = 0.0;
            double sum_weight = 0.0;

            for (const auto& p : kv.second) {
                sum_weighted_score += p.first;
                sum_weight += p.second;
            }

            if (sum_weight > 0 && rank_idx < 400) { // 配列外参照防止
                strength::cand_strength[rank_idx] = static_cast<float>(sum_weighted_score / sum_weight);
                std::cerr << "[" << rank_idx << "]:" << strength::cand_strength[rank_idx] << " ";
            }
        }
        std::cerr << std::endl;
    }

    std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Loading testing sgfs ..." << std::endl;
    std::string file_name = strength::testing_sgf_dir;

    std::cerr << "read: " << file_name << std::endl;
    std::vector<EnvironmentLoader> env_loaders = loadGames(file_name);

    std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Total loaded " << env_loaders.size() << " games" << std::endl;

    std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Running MCTS accuracy ..." << std::endl;
    STActorGroup ag;
    ag.initialize();
    std::vector<int> game_index(ag.getActors().size(), -1);
    std::vector<std::shared_ptr<actor::BaseActor>>& actors = ag.getActors();
    std::cerr << "--- DEBUG: Environment Identity Check ---" << std::endl;
    if (!actors.empty()) {
        // 1. 環境の名前を表示（これで "go" と出たらアウトです）
        std::cerr << "Env Name: " << actors[0]->getEnvironment().name() << std::endl;
        
        // 2. 盤面のサイズを表示（将棋なら9x9=81周辺、囲碁なら19x19=361など）
        std::cerr << "Board Size: " << actors[0]->getEnvironment().getBoardSize() << std::endl;
        
        // 3. アクションIDの最大値（Policy Size）を表示
        // ★ルール追加でここが変わっていないか要チェックです
        std::cerr << "Policy Size (Max Action ID): " << actors[0]->getEnvironment().getPolicySize() << std::endl;
    }
    std::cerr << "---------------------------------------" << std::endl;
    bool is_done = false;
    int current_game_index = 0;

    std::vector<int> mcts_correct(config::actor_num_simulation, 0);
    std::vector<std::vector<int>> ssa_correct_(temp_for_mcts_ssa_accuracy.size(), std::vector<int>(config::actor_num_simulation, 0));

    std::vector<int> total(config::actor_num_simulation, 0);
    while (!is_done) {
        is_done = true;
        for (size_t i = 0; i < actors.size(); ++i) {
            int move_number = actors[i]->getEnvironment().getActionHistory().size();
            if (game_index[i] != -1 && move_number < static_cast<int>(env_loaders[game_index[i]].getActionPairs().size())) {
                actors[i]->reset();
                is_done = false;
                for (int j = 0; j < move_number; ++j) { actors[i]->act(env_loaders[game_index[i]].getActionPairs()[j].first); }

            } else if (current_game_index < static_cast<int>(env_loaders.size())) {
                is_done = false;
                actors[i]->reset();
                game_index[i] = current_game_index++;
            } else {
                game_index[i] = -1;
                actors[i]->reset();
            }
        }
        if (is_done) { break; }
        ag.step();
        for (size_t i = 0; i < actors.size(); ++i) {
            if (game_index[i] == -1) { continue; }
            std::shared_ptr<StActor> actor = std::static_pointer_cast<StActor>(actors[i]);
            for (size_t j = 0; j < actor->getMCTSActionPerSimulation().size(); ++j) {
                const Action& mcts_action = actor->getMCTSActionPerSimulation()[j];
                const Action& sgf_action = env_loaders[game_index[i]].getActionPairs()[actors[i]->getEnvironment().getActionHistory().size() - 1].first;
                if (mcts_action.getActionID() == sgf_action.getActionID()) { mcts_correct[j]++; }
                for (size_t k = 0; k < temp_for_mcts_ssa_accuracy.size(); k++) {
                    const Action& ssa_action_ = actor->getSSAActionPerSimulation()[k][j];
                    if (ssa_action_.getActionID() == sgf_action.getActionID()) { ssa_correct_[k][j]++; }
                }
                total[j]++;
            }
        }
        for (size_t i = 0; i < total.size(); ++i) {
            int sim_count = i + 1;
            // 1回目、100の倍数、または最後の回数のみ出力
            if (sim_count == 1 || sim_count % 100 == 0 || sim_count == (int)total.size()) {
                std::cout << "simulation: " << sim_count
                          << ", mcts accuracy: " << (mcts_correct[i] * 100.0 / total[i]) << "% (" << mcts_correct[i] << "/" << total[i] << ")";

                for (size_t j = 0; j < temp_for_mcts_ssa_accuracy.size(); j++) {
                    std::cout << ", ssa_" << temp_for_mcts_ssa_accuracy[j] << " accuracy: " << (ssa_correct_[j][i] * 100.0 / total[i]) << "% (" << ssa_correct_[j][i] << "/" << total[i] << ")";
                }
                std::cout << std::endl;
            }
        }
        if (!total.empty() && total[0] >= 10000) {
            std::cerr << "Reached 10000 positions. Stopping..." << std::endl;
            // 最後に現在の精度を表示して終了
            for (size_t i = 0; i < total.size(); ++i) {
                int sim_count = i + 1;
                if (sim_count == 1 || sim_count % 100 == 0 || sim_count == (int)total.size()) {
                    std::cout << "FINAL simulation: " << sim_count
                              << ", mcts accuracy: " << (mcts_correct[i] * 100.0 / total[i]) << "% (" << mcts_correct[i] << "/" << total[i] << ")";
                    for (size_t j = 0; j < temp_for_mcts_ssa_accuracy.size(); j++) {
                        std::cout << ", ssa_" << temp_for_mcts_ssa_accuracy[j] << " accuracy: " << (ssa_correct_[j][i] * 100.0 / total[i]) << "% (" << ssa_correct_[j][i] << "/" << total[i] << ")";
                    }
                    std::cout << std::endl;
                }
            }
            break; // whileループを抜ける
        }
    }
    exit(0);
}
std::map<int, std::vector<std::pair<float, float>>> StModeHandler::calculatePosStrength(const std::vector<EnvironmentLoader>& env_loaders)
{
    if (env_loaders.empty()) { return {}; }
    std::vector<std::pair<float, float>> init(400, {0.0f, 0.0f});
    std::map<int, std::vector<std::pair<float, float>>> results;
    for (size_t i = 0; i < env_loaders.size(); ++i) {
        std::map<int, std::vector<std::pair<float, float>>> tmp = calculatePosStrength(env_loaders[i]);

        for (auto j : tmp) {
            if (results[j.first].size() == 0) results[j.first] = init;
            for (size_t k = 0; k < j.second.size(); k++) {
                results[j.first][k].first += j.second[k].first;
                results[j.first][k].second += j.second[k].second;
            }
        }
    }
    return results;
}
std::map<int, std::vector<std::pair<float, float>>> StModeHandler::calculatePosStrength(const EnvironmentLoader& env_loader)
{
    std::shared_ptr<StrengthNetwork> network = std::static_pointer_cast<StrengthNetwork>(network_);
    int count = 0;
    std::map<int, std::vector<std::pair<float, float>>> results;
    for (size_t pos = 0; pos < env_loader.getActionPairs().size(); ++pos) {
        Rotation rotation = static_cast<Rotation>(Random::randInt() % static_cast<int>(Rotation::kRotateSize));
        std::vector<float> features = calculateFeatures(env_loader, pos, rotation);
        network->pushBack(features);
        count++;
        if ((pos + 1) % 100 == 0 || pos == env_loader.getActionPairs().size() - 1) {
            std::vector<std::shared_ptr<network::NetworkOutput>> output = network->forward();
            for (int pos_ = 0; pos_ < count; ++pos_) {
                std::shared_ptr<StrengthNetworkOutput> s_output = std::static_pointer_cast<StrengthNetworkOutput>(output[pos_]);
                int rank = getRank(env_loader);
                if (strength::bt_use_weight) {
                    results[rank].push_back(std::make_pair(s_output->score_ * s_output->weight_, s_output->weight_));
                } else {
                    results[rank].push_back(std::make_pair(s_output->score_, 1));
                }
            }
            count = 0;
        }
    }
    return results;
}
std::string StModeHandler::getNetworkAbbeviation() const
{
    if (config::nn_type_name == "alphazero") {
        return "az";
    } else if (config::nn_type_name == "bt") {
        return "bt_b" + std::to_string(strength::bt_num_batch_size) + "_r" + std::to_string(strength::bt_num_rank_per_batch) + "_p" + std::to_string(strength::bt_num_position_per_rank);
    } else {
        return config::nn_type_name;
    }
}

} // namespace strength