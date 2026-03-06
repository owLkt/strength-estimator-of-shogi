#include <iostream>
#include "shogi.h" // あなたのshogi.hをインクルード

int main() {
    // ShogiEnvのインスタンスを作成
    // コンストラクタ内でreset()が呼ばれ、盤面が初期化されるはず
    minizero::env::shogi::ShogiEnv env;

    // toString()で初期盤面を表示
    std::cout << "--- Initial Board State ---" << std::endl;
    std::cout << env.toString() << std::endl;
    std::cout << "-------------------------" << std::endl;

    // getLegalActions()で合法手のリストを取得
    std::vector<minizero::env::shogi::ShogiAction> legal_actions = env.getLegalActions();

    // 合法手が存在するかチェック
    if (legal_actions.empty()) {
        std::cout << "No legal actions found. Test failed." << std::endl;
        return 1;
    }
    
    std::cout << "Found " << legal_actions.size() << " legal moves." << std::endl;
    std::cout << "Performing the first available legal move..." << std::endl;

    // 取得したリストの最初の合法手をact()で実行
    minizero::env::shogi::ShogiAction first_move = legal_actions[1];
    bool success = env.act(first_move);

    // 実行後の盤面をtoString()で表示し、着手が成功したか確認
    std::cout << "\n--- Board State After Move ---" << std::endl;
    if (success) {
        std::cout << "act() succeeded. Board should be updated." << std::endl;
    } else {
        std::cout << "act() failed." << std::endl;
    }
    std::cout << env.toString() << std::endl;
    std::cout << "----------------------------" << std::endl;

    return 0;
}