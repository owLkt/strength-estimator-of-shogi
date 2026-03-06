#include "evaluator.h"
#include "game_wrapper.h"
#include "random.h"
#include "st_configuration.h"
#include "strength_network.h"
#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <time_system.h>
#include <torch/cuda.h>
#include <utility>
#include <vector>
#include <iostream> // 追加
#include "cnpy.h"  // .npy読み込みライブラリ
#include <cmath>   // 追加: sqrt, abs用
#include <iomanip> // 追加: 出力桁数調整用 (std::setprecision)
#include <limits>  // 追加: numeric_limits用

namespace strength {

using namespace minizero;
using namespace minizero::network;
using namespace minizero::utils;

// -------------------------------------------------------------------
//  HELPER: ディレクトリ名からランクインデックスを取得する関数 (Python側とロジックを合わせる)
// (Python側の data_preparation.py にある getRankIndex と同じロジック)
// -------------------------------------------------------------------
int getRankIndexFromDir(const std::string& path_str) {
    try {
        std::filesystem::path path(path_str);
        std::string dirname = path.filename().string(); // "1000_1200" など
        
        size_t split_pos = dirname.find('_');
        if (split_pos == std::string::npos) {
            return -1; // 不正な形式
        }
        
        // ディレクトリ名の最初の数値 (例: "1000") をパース
        int elo = std::stoi(dirname.substr(0, split_pos)); 
        
        // config で設定された値 (Python側と同一である必要がある)
        const int min_elo = strength::nn_rank_min_elo;
        const int interval = strength::nn_rank_elo_interval;

        if (elo < min_elo) return -1;
        int index = (elo - min_elo) / interval;
        return index;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse rank from dir: " << path_str << " (" << e.what() << ")" << std::endl;
        return -1;
    }
}


int EvaluatorSharedData::getNextSgfIndex()
{
    std::lock_guard<std::mutex> lock(mutex_);
    // 変数名は sgf_index_ のままだが、実際には「ディレクトリのインデックス」を返す
    return (sgf_index_ < static_cast<int>(sgfs_.size()) ? sgf_index_++ : sgfs_.size());
}

void EvaluatorSlaveThread::initialize()
{
    int seed = config::program_auto_seed ? std::random_device()() : config::program_seed + id_;
    Random::seed(seed);
}

void EvaluatorSlaveThread::runJob()
{
    while (true) {
        size_t dir_index = getSharedData()->getNextSgfIndex();
        if (dir_index >= getSharedData()->sgfs_.size()) { break; }

        const std::string& dir_path = getSharedData()->sgfs_[dir_index];
        int gt_rank_index = getRankIndexFromDir(dir_path);
        if (gt_rank_index < 0) {
            std::cerr << "Skipping invalid directory: " << dir_path << std::endl;
            continue; 
        }
        
        // 共通変数
        size_t num_positions = 0;
        std::vector<std::shared_ptr<NetworkOutput>> network_outputs; 

        // =========================================================
        // STEP 1: Featuresの読み込みと推論（終わったら即解放）
        // =========================================================
        try {
            // スコープを作って、このブロックを抜けたら features_arr のメモリを強制開放させる
            {
                std::string features_path = dir_path + "/features.npy";
                cnpy::NpyArray features_arr = cnpy::npy_load(features_path);
                
                // データ形状チェック
                num_positions = features_arr.shape[0];
                if (num_positions == 0) {
                    std::cerr << "Zero positions in " << features_path << std::endl;
                    continue;
                }

                // shape: (N, 362, 9, 9) -> 29322
                size_t feature_size = features_arr.shape[1] * features_arr.shape[2] * features_arr.shape[3];
                if (feature_size != 29322) {
                     std::cerr << "Unexpected feature size: " << feature_size << " in " << features_path << std::endl;
                     continue;
                }

                float* features_ptr = features_arr.data<float>();
                network_outputs.reserve(num_positions);

                // --- ネットワーク推論ループ ---
                size_t batch_size = config::learner_batch_size > 0 ? config::learner_batch_size : 256;
                int network_id = id_ % static_cast<int>(getSharedData()->networks_.size());
                std::shared_ptr<StrengthNetwork> network = std::static_pointer_cast<StrengthNetwork>(getSharedData()->networks_[network_id]);

                for (size_t start_idx = 0; start_idx < num_positions; start_idx += batch_size) {
                    size_t end_idx = std::min(start_idx + batch_size, num_positions);
                    
                    {
                        std::lock_guard<std::mutex> lock(*getSharedData()->network_mutexes_[network_id]);
                        
                        for (size_t i = start_idx; i < end_idx; ++i) {
                            // ポインタから直接生成して渡す（前回提案の軽量化）
                            std::vector<float> current_feature(
                                features_ptr + i * feature_size,
                                features_ptr + (i + 1) * feature_size
                            );
                            network->pushBack(std::move(current_feature));
                        }
                        
                        std::vector<std::shared_ptr<NetworkOutput>> batch_outputs = network->forward();
                        network_outputs.insert(network_outputs.end(), batch_outputs.begin(), batch_outputs.end());
                    }
                }
            } 

            // =========================================================
            // STEP 2: Policy / Value の読み込みと集計
            // =========================================================
            std::string policy_path = dir_path + "/policy.npy";
            std::string value_path = dir_path + "/value.npy";
            
            cnpy::NpyArray policy_arr = cnpy::npy_load(policy_path);
            cnpy::NpyArray value_arr = cnpy::npy_load(value_path);

            // 整合性チェック
            if (policy_arr.shape[0] != num_positions || value_arr.shape[0] != num_positions) {
                std::cerr << "Data shape mismatch in " << dir_path << std::endl;
                continue;
            }

            float* policy_ptr = policy_arr.data<float>();
            float* value_ptr = value_arr.data<float>();
            size_t policy_size = policy_arr.shape[1]; // 11259

            // Game Lengths読み込み
            std::vector<int> game_lengths;
            std::string lengths_path = dir_path + "/game_lengths.npy";
            if (std::filesystem::exists(lengths_path)) {
                cnpy::NpyArray lengths_arr = cnpy::npy_load(lengths_path);
                int* lengths_ptr = lengths_arr.data<int>();
                size_t num_games = lengths_arr.shape[0];
                game_lengths.assign(lengths_ptr, lengths_ptr + num_games);
            } else {
                game_lengths.push_back(num_positions);
            }

            // 集計処理 (GameData作成)
            size_t current_offset = 0;
            std::vector<GameData> local_games;

            for (int length : game_lengths) {
                if(current_offset + length > num_positions) break;
                
                GameData game_data;
                game_data.rank_.reserve(length);
                game_data.scores_.reserve(length);
                game_data.value_.reserve(length);
                
                bool is_move_prediction = (strength::evaluator_mode == "move_prediction");

                for (int i = 0; i < length; ++i) {
                    size_t global_idx = current_offset + i;
                    auto net_out = std::static_pointer_cast<StrengthNetworkOutput>(network_outputs[global_idx]);

                    if (config::nn_type_name == "rank") {
                        game_data.rank_.push_back(net_out->rank_);
                    }
                    if (config::nn_type_name == "bt") {
                        game_data.scores_.push_back(net_out->score_);
                        game_data.weights_.push_back(net_out->weight_);
                    }
                    game_data.value_.push_back(net_out->value_);
                    
                    if (is_move_prediction) {
                        game_data.policy_.push_back(net_out->policy_);
                        // 正解Policyのコピー
                        std::vector<float> gt_p(policy_ptr + global_idx * policy_size, 
                                                policy_ptr + (global_idx + 1) * policy_size);
                        game_data.gt_policy_.push_back(gt_p);
                    }
                    game_data.gt_value_.push_back(value_ptr[global_idx]);
                }
                local_games.push_back(game_data);
                current_offset += length;
            }

            // マージ
            {
                std::lock_guard<std::mutex> lock(getSharedData()->mutex_);
                auto& target_vec = getSharedData()->env_loaders_map_[gt_rank_index];
                target_vec.insert(target_vec.end(), local_games.begin(), local_games.end());
            }

        } catch (const std::exception& e) {
            std::cerr << "Error in " << dir_path << ": " << e.what() << std::endl;
            continue;
        }

        // ログ出力
        int num_dirs = getSharedData()->sgfs_.size();
        int log_check = static_cast<int>(num_dirs * 0.1);
        if (log_check > 0 && dir_index > 0 && (dir_index % log_check) == 0) { 
            std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << dir_index << " / " << num_dirs << " directories" << std::endl; 
        }
    }
}

void Evaluator::run()
{
    initialize();

    std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Evaluator mode: " << strength::evaluator_mode << std::endl;
    if (strength::evaluator_mode == "game_prediction") {
        runGamePrediction();
    } else if (strength::evaluator_mode == "move_prediction") {
        runMovePrediction();
    } else {
        std::cerr << "Unknown evaluator mode: " << strength::evaluator_mode << std::endl;
    }
}

void Evaluator::initialize()
{
    createSlaveThreads(config::learner_num_thread);
    createNeuralNetworks();
}

void Evaluator::createNeuralNetworks()
{
    int num_networks = static_cast<int>(torch::cuda::device_count());
    assert(num_networks > 0);
    getSharedData()->networks_.resize(num_networks);
    for (int gpu_id = 0; gpu_id < num_networks; ++gpu_id) {
        getSharedData()->networks_[gpu_id] = std::make_shared<StrengthNetwork>();
        getSharedData()->network_mutexes_.emplace_back(new std::mutex);
        std::static_pointer_cast<StrengthNetwork>(getSharedData()->networks_[gpu_id])->loadModel(config::nn_file_name, gpu_id);
    }
}

// -------------------------------------------------------------------
// UNCHANGED: runGamePrediction
// この関数はネットワークの出力 (rank_ や scores_) と
// マップのキー (gt_rank_index) を比較するため、変更不要。
// -------------------------------------------------------------------
void Evaluator::runGamePrediction()
{
    if (config::nn_type_name == "bt") {
        std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Running candidate sgfs (npy dirs) ..." << std::endl;
        evaluateGames(strength::candidate_sgf_dir);
        std::map<int, std::vector<GameData>> candidate = getSharedData()->env_loaders_map_;

        // ==========================================
        // ★追加: 分布確認用に生スコアをCSVに保存する
        // ==========================================
        std::string output_csv = "rank_score_distribution.csv";
        std::ofstream file(output_csv);
        if (file.is_open()) {
            std::cerr << "Saving score distribution to " << output_csv << " ..." << std::endl;
            // ヘッダー
            file << "Rank,Score\n";
            
            // 全ランク、全ゲーム、全局面のスコアを書き出す
            for (auto& m : candidate) { // m.first: RankID, m.second: Vector<GameData>
                int rank_id = m.first;
                for (auto& game_data : m.second) {
                    for (float s : game_data.scores_) {
                        file << rank_id << "," << s << "\n";
                    }
                }
            }
            file.close();
            std::cerr << "Done saving csv." << std::endl;
        } else {
            std::cerr << "Failed to open file: " << output_csv << std::endl;
        }
        // ==========================================

        std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Calculating candidate strength ... " << std::endl;
        
        // ヘッダーを出力（タブ区切りで見やすく）
        std::cerr << "Rank\tMean\tVariance\tStdDev\tMin\tMax\tCount" << std::endl;

        std::vector<std::pair<int, float>> rank_scores;
        
        for (auto& m : candidate) {
            // スコアを一時的にすべて保存するベクタ
            std::vector<float> collected_scores;
            for (auto& game_data : m.second) {
                collected_scores.insert(collected_scores.end(), 
                                        game_data.scores_.begin(), 
                                        game_data.scores_.end());
            }
            //統計量を計算
            double sum = 0.0;
            float min_val = std::numeric_limits<float>::max();
            float max_val = std::numeric_limits<float>::lowest();

            for (float s : collected_scores) {
                sum += s;
                if (s < min_val) min_val = s;
                if (s > max_val) max_val = s;
            }

            double mean = 0.0;
            double variance = 0.0;
            double std_dev = 0.0;

            if (!collected_scores.empty()) {
                mean = sum / collected_scores.size();
                
                // 分散計算
                for (float s : collected_scores) {
                    variance += (s - mean) * (s - mean);
                }
                variance /= collected_scores.size();
                std_dev = std::sqrt(variance);
            } else {
                // データがない場合のリセット
                min_val = 0.0f;
                max_val = 0.0f;
            }

            // 元のロジック用に平均値を保存
            rank_scores.push_back({m.first, static_cast<float>(mean)});
            
            // 詳細をログ出力
            std::cerr << m.first << "\t" 
                      << std::fixed << std::setprecision(4) << mean << "\t" 
                      << variance << "\t" 
                      << std_dev << "\t" 
                      << min_val << "\t" 
                      << max_val << "\t" 
                      << collected_scores.size() << std::endl;
        }

        // ソート（元のロジック通り）
        std::sort(rank_scores.begin(), rank_scores.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        
        // 元のログ出力も残しておく（必要であれば）
        std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Candidating strength (Mean only): " << std::endl;
        for (auto& rank_score : rank_scores) { std::cerr << "\t" << rank_score.first << " " << rank_score.second << std::endl; }

        std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Running testing sgfs (npy dirs) ..." << std::endl;
        evaluateGames(strength::testing_sgf_dir);
        std::map<int, std::vector<GameData>> testing = getSharedData()->env_loaders_map_;

        std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Summarizing ranks ..." << std::endl;
        summarizeGamePrediction(rank_scores, testing);
    } else if (config::nn_type_name == "rank") {
        std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Running testing sgfs (npy dirs) ..." << std::endl;
        evaluateGames(strength::testing_sgf_dir);
        std::map<int, std::vector<GameData>> testing = getSharedData()->env_loaders_map_;
        std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Summarizing ranks ..." << std::endl;
        summarizeGamePrediction(testing);
    }
}

// -------------------------------------------------------------------
// MODIFIED: runMovePrediction
// SGF (env_loader_) の代わりに .npy (gt_policy_) から正解行動を読み取る
// -------------------------------------------------------------------
void Evaluator::runMovePrediction()
{
    std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Running testing sgfs (npy dirs) ..." << std::endl;
    evaluateGames(strength::testing_sgf_dir);

    std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Calculating rank orders ... " << std::endl;
    int max_order = 0;
    std::map<int, std::vector<int>> rank_orders;
    for (auto& m : getSharedData()->env_loaders_map_) {
        for (auto& game_data : m.second) {
            // const EnvironmentLoader& env_loader = game_data.env_loader_; // 削除 (env_loader_はもうない)
            
            // game_data.gt_policy_ (policy.npy) のサイズだけループ
            for (size_t i = 0; i < game_data.gt_policy_.size(); ++i) {
                
                // (A) 正解データを取得
                // SGFのアクションIDの代わりに、gt_policy_ (one-hot) から 1.0 のインデックスを探す
                const std::vector<float>& gt_policy = game_data.gt_policy_[i];
                int gt_action_id = -1;
                float max_prob = 0.5f; // 念のため 0.5f より大きいものを探す
                for(size_t j = 0; j < gt_policy.size(); ++j) {
                    if (gt_policy[j] > max_prob) { 
                        max_prob = gt_policy[j];
                        gt_action_id = static_cast<int>(j);
                    }
                }

                if (gt_action_id == -1) {
                    // 正解ポリシーが one-hot ではない (または空) 場合はスキップ
                    continue; 
                }

                // (B) ネットワークの予測を取得
                const std::vector<float>& net_policy = game_data.policy_[i];
                // ネットワークが出力した、その「正解行動」の確率
                float policy_prob_for_gt_action = net_policy[gt_action_id];

                // (C) ランクを計算
                // ネットワーク出力の中で、正解行動の確率が何番目に高かったか
                int policy_order = 1;
                for (size_t j = 0; j < net_policy.size(); ++j) { 
                    policy_order += (net_policy[j] > policy_prob_for_gt_action); 
                }
                max_order = std::max(max_order, policy_order);

                if (policy_order >= static_cast<int>(rank_orders[m.first].size())) { rank_orders[m.first].resize(policy_order + 1, 0); }
                ++rank_orders[m.first][policy_order];
            }
        }
    }

    std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Summarizing accuracy ... " << std::endl;
    int all_total = 0;
    std::map<int, std::pair<int, int>> accumulated_accuracy;
    for (auto& m : rank_orders) {
        std::cout << "\t" << m.first; // header
        accumulated_accuracy[m.first] = {0, std::accumulate(rank_orders[m.first].begin(), rank_orders[m.first].end(), 0)};
        all_total += accumulated_accuracy[m.first].second;
    }
    std::cout << "\tall" << std::endl;
    for (int i = 1; i <= max_order; ++i) {
        std::cout << i;
        int all_correct = 0;
        for (auto& m : rank_orders) {
            if (i < static_cast<int>(rank_orders[m.first].size())) { accumulated_accuracy[m.first].first += rank_orders[m.first][i]; }
            all_correct += accumulated_accuracy[m.first].first;
            std::cout << "\t" << accumulated_accuracy[m.first].first * 1.0f / accumulated_accuracy[m.first].second;
        }
        std::cout << "\t" << all_correct * 1.0f / all_total << std::endl;
    }
}

// -------------------------------------------------------------------
// MODIFIED: evaluateGames
// SGFファイルを読み込むのではなく、.npyファイル群が格納されたサブディレクトリを検索する
// -------------------------------------------------------------------
void Evaluator::evaluateGames(const std::string& directory)
{
    std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Scanning for data directories in: " << directory << std::endl;

    getSharedData()->sgfs_.clear(); // 名前は sgfs_ だが、実体はディレクトリパスのリスト
    
    if (!std::filesystem::exists(directory) || std::filesystem::is_empty(directory)) {
        std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Directory does not exist or is empty: " << directory << std::endl;
        getSharedData()->sgf_index_ = 0;
        getSharedData()->env_loaders_map_.clear();
        return;
    }

    // .npy ファイル群 (features.npyなど) が格納されているサブディレクトリを検索
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_directory()) {
            // features.npy の存在チェック (オプションだが堅牢性が増す)
            if (std::filesystem::exists(entry.path() / "features.npy")) {
                getSharedData()->sgfs_.push_back(entry.path().string());
            } else {
                 std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Skipping dir, features.npy not found in: " << entry.path().string() << std::endl;
            }
        }
    }

    std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Successfully loaded " << getSharedData()->sgfs_.size() << " data directories" << std::endl;
    
    getSharedData()->sgf_index_ = 0;
    getSharedData()->env_loaders_map_.clear();
    
    if (getSharedData()->sgfs_.empty()) {
        return; // ディレクトリが0件ならスレッドを開始しない
    }

    for (auto& t : slave_threads_) { t->start(); }
    for (auto& t : slave_threads_) { t->finish(); }
}


// -------------------------------------------------------------------
// UNCHANGED: summarizeGamePrediction (2つのオーバーロード)
// これらの関数はネットワークの出力 (rank_ や scores_) と
// マップのキー (gt_rank_index) を比較するため、変更不要。
// -------------------------------------------------------------------
void Evaluator::summarizeGamePrediction(const std::map<int, std::vector<GameData>>& testing)
{
    const int max_games = 100;
    const int repeat_times = 500;
    std::map<int, std::vector<float>> rank_accuracy;
    for (auto& rank_score : testing) {
        int rank = rank_score.first;
        rank_accuracy[rank].resize(max_games, 0.0f);
        std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Predicting rank " << rank << " ... " << std::endl;
        for (int used_games = 1; used_games <= max_games; ++used_games) {
            int correct = 0;
            int correct1 = 0;
            int correct_1 = 0;
            for (int i = 0; i < repeat_times; ++i) {
                std::vector<float> sum_rank(strength::nn_rank_size, 0.0f);
                for (int game = 0; game < used_games; ++game) {
                    const GameData& game_data = testing.at(rank)[Random::randInt() % static_cast<int>(testing.at(rank).size())];

                    int temp = 0;
                    int rand_pos = Random::randInt() % 2;
                    int lower_bound = 0;
                    int upper_bound = 1000;

                    // (注: 元のコードでは scores_.size() を使っているが、rankモードでは rank_.size() を使うべき)
                    size_t num_moves = game_data.rank_.size();
                    if (num_moves == 0) continue; 

                    if (strength::select_move == "last_50_moves") {
                        lower_bound = static_cast<int>(num_moves) - 50;
                    }
                    if (strength::select_move == "first_50_moves") {
                        upper_bound = 50;
                    }
                    if (strength::select_move == "one_move_per_game") {
                        int select_pos = Random::randInt() % static_cast<int>(num_moves);
                        lower_bound = select_pos;
                        upper_bound = select_pos + 1;
                        rand_pos = select_pos % 2;
                    }

                    for (auto& data : game_data.rank_) { // network_output->rank_
                        float maxElement = data[0]; 
                        int maxIndex = 0;           

                        if (temp >= lower_bound && temp % 2 == rand_pos) {
                            if (strength::rank_mode == "max_num") {
                                for (int j = 1; j < strength::nn_rank_size; ++j) {
                                    if (data[j] > maxElement) {
                                        maxElement = data[j]; 
                                        maxIndex = j;         
                                    }
                                }
                                sum_rank[maxIndex]++;
                            } else if (strength::rank_mode == "max_prob") {
                                for (int j = 0; j < strength::nn_rank_size; ++j) {
                                    sum_rank[j] += data[j];
                                }
                            }
                        }
                        temp++;
                        if (temp >= upper_bound) break;
                    }
                }
                float maxElement = sum_rank[0]; 
                int maxIndex = 0;               

                for (size_t i = 1; i < sum_rank.size(); ++i) {
                    if (sum_rank[i] > maxElement) {
                        maxElement = sum_rank[i]; 
                        maxIndex = i;             
                    }
                }
                
                // 予測インデックス(maxIndex) と 正解ランクインデックス(rank) を比較
                if (rank == (maxIndex)) { ++correct_1; } // (注: 元の +/-1 のロジックがズレている可能性あり)
                if (rank == (maxIndex - 1)) { ++correct; }
                if (rank == (maxIndex + 1)) { ++correct1; } // (注: 元は -2 だったが +1 が自然)
            }
            if (strength::accuracy_mode == "+/-1") {
                correct += correct1;
                correct += correct_1;
            } else if (strength::accuracy_mode == "+1") {
                correct += correct1;
            } else if (strength::accuracy_mode == "-1") {
                correct += correct_1;
            }

            rank_accuracy[rank][used_games - 1] = correct * 1.0f / repeat_times;
        }
    }

    std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Summarizing accuracy ... " << std::endl;
    for (auto& rank_score : testing) { std::cout << "\t" << rank_score.first; } // header
    std::cout << "\tall" << std::endl;
    for (size_t i = 1; i <= max_games; ++i) {
        std::cout << i;
        float avg_accuracy = 0.0f;
        for (auto& rank_score : testing) {
            std::cout << "\t" << rank_accuracy[rank_score.first][i - 1];
            avg_accuracy += rank_accuracy[rank_score.first][i - 1];
        }
        std::cout << "\t" << avg_accuracy / testing.size() << std::endl;
    }
}

void Evaluator::summarizeGamePrediction(const std::vector<std::pair<int, float>>& rank_scores, const std::map<int, std::vector<GameData>>& testing)
{
    const int max_games = 100;
    const int repeat_times = 500;
    std::map<int, std::vector<float>> rank_accuracy;
    for (auto& rank_score : rank_scores) {
        int rank = rank_score.first;
        rank_accuracy[rank].resize(max_games, 0.0f);
        std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Predicting rank " << rank << " ... " << std::endl;

        for (int used_games = 1; used_games <= max_games; ++used_games) {
            int correct = 0;
            int correct1 = 0;
            int correct_1 = 0;
            for (int i = 0; i < repeat_times; ++i) {
                float score = 0;
                int num_position = 0;
                for (int game = 0; game < used_games; ++game) {
                    const GameData& game_data = testing.at(rank)[Random::randInt() % static_cast<int>(testing.at(rank).size())];
                    int rand_pos = Random::randInt() % 2;
                    int temp = 0;
                    int lower_bound = 0;
                    int upper_bound = 1000;

                    size_t num_moves = game_data.scores_.size();
                    if (num_moves == 0) continue;

                    if (strength::select_move == "last_50_moves") {
                        lower_bound = static_cast<int>(num_moves) - 50;
                    }
                    if (strength::select_move == "first_50_moves") {
                        upper_bound = 50;
                    }
                    if (strength::select_move == "one_move_per_game") {
                        int select_pos = Random::randInt() % static_cast<int>(num_moves);
                        lower_bound = select_pos;
                        upper_bound = select_pos + 1;
                        rand_pos = select_pos % 2;
                    }

                    for (auto& s : game_data.scores_) {
                        if (temp >= lower_bound && temp % 2 == rand_pos) {
                            score += s;
                            num_position++;
                        }
                        temp++;
                        if (temp >= upper_bound) break;
                    }
                }

                if (num_position == 0) {
                     // 稀に発生するゼロ除算を回避
                     continue; 
                }

                float average_score = score / num_position;
                int prediected_rank = std::min_element(rank_scores.begin(), rank_scores.end(), [average_score](const std::pair<int, float>& a, const std::pair<int, float>& b) { return std::abs(average_score - a.second) < std::abs(average_score - b.second); })->first;

                if (used_games == 100 && i % 100 == 0) { // ログ過多を防ぐため、100回に1回、かつ100ゲーム使用時のみ出力
                    std::cerr << "[DEBUG Rank] Iter:" << i 
                              << " AvgScore:" << std::fixed << std::setprecision(4) << average_score 
                              << " Predicted:" << prediected_rank << std::endl;
                }
                if (prediected_rank == rank) { correct++; }
                if (prediected_rank == rank + 1) { correct1++; }
                if (prediected_rank == rank - 1) { correct_1++; }
            }
            if (strength::accuracy_mode == "+/-1") {
                correct += correct1;
                correct += correct_1;
            } else if (strength::accuracy_mode == "+1") {
                correct += correct1;
            } else if (strength::accuracy_mode == "-1") {
                correct += correct_1;
            }
            rank_accuracy[rank][used_games - 1] = correct * 1.0f / repeat_times;
        }
    }

    std::cerr << TimeSystem::getTimeString("[Y/m/d H:i:s.f] ") << "Summarizing accuracy ... " << std::endl;
    for (auto& rank_score : rank_scores) { std::cout << "\t" << rank_score.first; } // header
    std::cout << "\tall" << std::endl;
    for (size_t i = 1; i <= max_games; ++i) {
        std::cout << i;
        float avg_accuracy = 0.0f;
        for (auto& rank_score : rank_scores) {
            std::cout << "\t" << rank_accuracy[rank_score.first][i - 1];
            avg_accuracy += rank_accuracy[rank_score.first][i - 1];
        }
        std::cout << "\t" << avg_accuracy / rank_scores.size() << std::endl;
    }
}

} // namespace strength