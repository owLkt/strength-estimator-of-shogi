#include "game_wrapper.h"
#include "sgf_loader.h"
#include "st_configuration.h"
#include <algorithm>
#include <string>
#include <vector>
#include <iostream>

namespace strength {

using namespace minizero;
using namespace minizero::utils;
std::vector<EnvironmentLoader> loadGames(const std::string& file_name)
{
    std::vector<EnvironmentLoader> env_loaders;
    std::ifstream fin(file_name, std::ifstream::in);

    // ★追加: ファイルが開けたかチェック
    if (!fin.is_open()) {
        std::cerr << "[Error] Cannot open file: " << file_name << std::endl;
        std::cerr << "Current working directory may differ from where the file is." << std::endl;
        return env_loaders;
    } else {
        std::cerr << "[Info] File opened successfully. Reading lines..." << std::endl;
    }

    int line_count = 0;
    for (std::string content; std::getline(fin, content);) {
        line_count++;
        // ★追加: 空行チェック
        if (content.empty()) {
            std::cerr << "[Warning] Line " << line_count << " is empty." << std::endl;
            continue;
        }

        EnvironmentLoader env_loader = loadGame(content);
        
        // ★追加: ロード結果のチェック
        if (env_loader.getActionPairs().empty()) { 
            std::cerr << "[Warning] Line " << line_count << ": Parsed but no actions found (loadFromString failed or empty game)." << std::endl;
            // デバッグ用に先頭部分を表示
            std::cerr << "   Line content head: " << content.substr(0, 50) << "..." << std::endl;
            continue; 
        } else {
             // 成功した場合も一応ログ（量が多い場合はコメントアウト）
             // std::cerr << "[Info] Line " << line_count << ": Loaded " << env_loader.getActionPairs().size() << " moves." << std::endl;
        }

        env_loaders.push_back(env_loader);
    }
    
    std::cerr << "[Info] Total lines read: " << line_count << std::endl;
    return env_loaders;
}
EnvironmentLoader loadGame(const std::string& file_content)
{
#if GO
    EnvironmentLoader env_loader;
    // if (env_loader.loadFromString(file_content)) { return env_loader; }

    if (file_content.empty()) { return EnvironmentLoader(); }
    if (file_content.find("(") == std::string::npos) { return EnvironmentLoader(); }

    SGFLoader sgf_loader;
    if (!sgf_loader.loadFromString(file_content)) { return EnvironmentLoader(); }
    if (std::stoi(sgf_loader.getTags().at("SZ")) != 19) { return EnvironmentLoader(); }

    env_loader.reset();
    env_loader.addTag("SZ", sgf_loader.getTags().at("SZ"));
    env_loader.addTag("KM", sgf_loader.getTags().at("KM"));
    env_loader.addTag("RE", std::to_string(sgf_loader.getTags().at("RE")[0] == 'B' ? 1.0f : -1.0f));
    env_loader.addTag("PB", sgf_loader.getTags().at("PB"));
    env_loader.addTag("PW", sgf_loader.getTags().at("PW"));
    env_loader.addTag("BR", sgf_loader.getTags().at("BR"));
    env_loader.addTag("WR", sgf_loader.getTags().at("WR"));
    for (auto& action_string : sgf_loader.getActions()) { env_loader.addActionPair(Action(action_string.first, std::stoi(sgf_loader.getTags().at("SZ"))), action_string.second); }
    return env_loader;
#else
    EnvironmentLoader env_loader;
    env_loader.loadFromString(file_content);
    return env_loader;
#endif
}

std::vector<float> calculateFeatures(const Environment& env, minizero::utils::Rotation rotation /* = minizero::utils::Rotation::kRotationNone */)
{
    return env.getFeatures(rotation);
}

std::vector<float> calculateFeatures(const EnvironmentLoader& env_loader, const int& pos, minizero::utils::Rotation rotation /* = minizero::utils::Rotation::kRotationNone */)
{
    return env_loader.getFeatures(pos, rotation);
}

int getRank(const EnvironmentLoader& env_loader)
{
#if GO
    std::string rank_str = env_loader.getTag("BR");
    if (rank_str.empty()) { return 0; }
    int rank = std::stoi(rank_str);
    if (rank_str.find_first_of("kK") != std::string::npos) {
        rank = -rank + 1;

        // 3-5k,1-2k,1-9D
        if (rank <= -2)
            rank = -1;
        else if (rank <= 0)
            rank = 0;
    }
    return rank;
#elif CHESS
    // 0 under min elo
    // 1 ~ rank size - 2: min elo + interval * rank
    // rank size - 1: above max elo
    std::string rank_str = env_loader.getTag("BR");
    int rank = std::stoi(rank_str);
    int ret = (rank - strength::nn_rank_min_elo) / strength::nn_rank_elo_interval;
    return ret;
#elif SHOGI
    // ★ここを修正（SHOGI用）
    
    // 1. shogi.cpp が black_rate を "BR" に変換してくれているので、それを取得
    std::string rank_str = env_loader.getTag("BR");
    
    // データがない場合は 0 を返す
    if (rank_str.empty()) { return 0; }
    
    // 2. 文字列 "4474.0" を数値に変換
    // .0 がついているので、一度 float にしてから int にキャストするのが安全です
    float rating_f = std::stof(rank_str);
    int rating = static_cast<int>(rating_f);
    
    // 3. ランクのインデックス計算
    // Python側の設定 (min_elo=3100, interval=200) に合わせる
    // 定数が定義されていれば strength::nn_rank_min_elo を使ってください
    // 定義が不明な場合は、Pythonコードに合わせて直接数値を書きます：
    int min_elo = 1500;
    int interval = 400;
    
    int ret = (rating - min_elo) / interval;
    
    // 念のための範囲チェック（負の数にならないように）
    if (ret < 0) ret = 0;
    
    return ret;

#else
        // TODO: each game should implement its own getRank
    return 0;
#endif
}

} // namespace strength
