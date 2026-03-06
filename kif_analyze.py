import re
import glob
import os
import pandas as pd

def parse_multi_session_kif(directory_path):
    abs_path = os.path.abspath(directory_path)
    file_list = glob.glob(os.path.join(abs_path, "*.kif"))
    if not file_list: return pd.DataFrame()
    
    all_game_data = []
    for file_path in file_list:
        filename = os.path.basename(file_path)
        creation_time = os.path.getctime(file_path)
        try:
            with open(file_path, 'r', encoding='cp932', errors='ignore') as f:
                content = f.read()
            
            # 1. プレイヤー名の抽出
            black = re.search(r'^先手：(.+)', content, re.MULTILINE)
            white = re.search(r'^後手：(.+)', content, re.MULTILINE)
            black_player = black.group(1).strip() if black else "Unknown"
            white_player = white.group(1).strip() if white else "Unknown"

            # 2. 勝敗判定（ロジックの整理）
            winner_side = "Unknown"
            last_move_num = 0

            # 末尾の手数を取得（共通）
            moves = re.findall(r'^\s*(\d+)\s+', content, re.MULTILINE)
            last_move_num = int(moves[-1]) if moves else 0

            if "持将棋" in content or "千日手" in content:
                winner_side = "Draw" # 明示的にDrawとする
            elif "投了" in content or "先手勝ち" in content or "後手勝ち" in content or "の勝ち" in content:
                # 投了などの場合、手数で判定（投了した側の手数は含まれないため、偶数なら先手投了＝後手勝ち）
                # ※「196 投了」なら実質195手完。
                if "投了" in content:
                    actual_moves = last_move_num - 1
                else:
                    actual_moves = last_move_num
                winner_side = "White" if actual_moves % 2 == 0 else "Black"
            
            all_game_data.append({
                'CreationTime': creation_time,
                'Black': black_player,
                'White': white_player,
                'TotalMoves': last_move_num,
                'WinnerSide': winner_side,
                'FileName': filename
            })
        except Exception as e:
            print(f"読み込み失敗 ({filename}): {e}")

    df = pd.DataFrame(all_game_data)
    if not df.empty:
        df = df.sort_values('CreationTime').reset_index(drop=True)
    return df

def analyze_specific_matchup(df, engine_1, engine_2):
    if df.empty:
        print("データが空です。")
        return None

    # 指定ペアの抽出
    mask = (df['Black'].str.contains(engine_1) | df['White'].str.contains(engine_1)) & \
           (df['Black'].str.contains(engine_2) | df['White'].str.contains(engine_2))
    target_df = df[mask].copy()

    if target_df.empty:
        print(f"指定されたペア ({engine_1} vs {engine_2}) の対局は見つかりませんでした。")
        return None

    # 時系列ソートと100局ごとのSessionID
    target_df = target_df.sort_values('CreationTime').reset_index(drop=True)
    target_df['SessionID'] = (target_df.index // 100) + 1
    
    # engine_1を基準とした勝敗判定
    def judge(row):
        is_e1_black = engine_1 in row['Black']
        if (is_e1_black and row['WinnerSide'] == "Black") or (not is_e1_black and row['WinnerSide'] == "White"):
            return "Win"
        return "Loss"

    target_df['Result'] = target_df.apply(judge, axis=1)
    target_df['Opponent'] = target_df.apply(lambda r: r['White'] if engine_1 in r['Black'] else r['Black'], axis=1)

    # 集計 (GameNumをFileNameに変更してエラー回避)
    summary = target_df.groupby(['SessionID', 'Opponent']).agg(
        対局数=('FileName', 'count'),
        勝利=('Result', lambda x: (x == 'Win').sum()),
        敗北=('Result', lambda x: (x == 'Loss').sum()),
        平均手数=('TotalMoves', 'mean'),
    ).reset_index()
    
    summary['勝率'] = (summary['勝利'] / summary['対局数'] * 100).round(1).astype(str) + '%'
    
    print(f"\n### {engine_1} から見た対戦成績 (相手キーワード: {engine_2}) ###")
    print(summary.to_string(index=False))
    return summary
def analyze_self_matchup(df, engine_name, copy_label):
    # 1. MiniZero同士の対局に絞り込む
    mask = df['Black'].str.contains(engine_name) & df['White'].str.contains(engine_name)
    target_df = df[mask].copy()
    
    if target_df.empty:
        return None

    # 2. ファイル名から連番を抽出
    def extract_game_index(filename):
        match = re.search(r'連続対局\s+(\d+)_', filename)
        return int(match.group(1)) if match else None

    target_df['GameIndex'] = target_df['FileName'].apply(extract_game_index)

    # 3. 作成日時で厳密にソート（これが全ての基準）
    target_df = target_df.sort_values('CreationTime').reset_index(drop=True)

    # 4. 【重要】SessionIDの割り当て（重複・誤判定防止）
    session_ids = []
    current_session = 1
    prev_idx = -1
    
    for idx in target_df['GameIndex']:
        if idx is not None:
            # 「1番」が来た、または「前の番号より小さい」場合に新しいセッションとみなす
            if idx == 1 or (prev_idx != -1 and idx < prev_idx):
                if prev_idx != -1: # 初回ループを除外
                    current_session += 1
            prev_idx = idx
        session_ids.append(current_session)
        
    target_df['SessionID'] = session_ids

    # 5. 勝敗判定
    def assign_roles(row):
        if row['WinnerSide'] == "Draw": return "Draw"
        is_black_copy = copy_label in row['Black']
        if is_black_copy:
            return "CopyWins" if row['WinnerSide'] == "Black" else "OrigWins"
        else:
            return "OrigWins" if row['WinnerSide'] == "Black" else "CopyWins"

    target_df['WinnerRole'] = target_df.apply(assign_roles, axis=1)

    # 6. 集計
    summary = target_df.groupby('SessionID').agg(
        対局数=('FileName', 'count'),
        オリ勝利=('WinnerRole', lambda x: (x == 'OrigWins').sum()),
        コピー勝利=('WinnerRole', lambda x: (x == 'CopyWins').sum()),
        引き分け=('WinnerRole', lambda x: (x == 'Draw').sum()),
        平均手数=('TotalMoves', 'mean'),
        開始時刻=('CreationTime', lambda x: pd.to_datetime(x.min(), unit='s').strftime('%H:%M')),
        終了時刻=('CreationTime', lambda x: pd.to_datetime(x.max(), unit='s').strftime('%H:%M'))
    ).reset_index()

    # 勝率
    def calc_win_rate(win, loss):
        total = win + loss
        return f"{(win / total * 100):.1f}%" if total > 0 else "0.0%"

    summary['オリ勝率'] = summary.apply(lambda r: calc_win_rate(r['オリ勝利'], r['コピー勝利']), axis=1)
    summary['コピー勝率'] = summary.apply(lambda r: calc_win_rate(r['コピー勝利'], r['オリ勝利']), axis=1)

    print(f"\n### オリジナル進化測定結果 (セッション区切り・時刻付き) ###")
    print(summary.to_string(index=False))
    return summary

# 実行
df = parse_multi_session_kif("../ドキュメント/ShogiHome")
if not df.empty:
    print(f"{len(df)} 局のデータを読み込みました。")
    # AobaZeroを軸に、MiniZeroとの対局を集計
    #analyze_specific_matchup(df, "MiniZero", "AobaZero")
    # MiniZero同士の対局を集計したい場合はこちら
    analyze_self_matchup(df, "MiniZero", "コピー")