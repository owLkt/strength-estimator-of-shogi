#!/bin/bash
# ホスト側から Docker コンテナ内のエンジンを呼び出す
# -i はインタラクティブモード（標準入力を受け付けるため）
# "$@" を付けることで、ShogiHome側で設定した引数をすべてコンテナ内に渡す
CONF_PATH="/workspace/cfg/se_shogi_mcts.cfg"
docker exec -i boring_torvalds /workspace/build/shogi/shogi_usi -conf "$CONF_PATH" "$@"