import re
import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt

def generate_rank_accuracy_report(log_dir, target_sim):
    """
    指定したシミュレーション回数のデータを抽出し、ヒートマップを作成する
    """
    all_data = []
    file_paths = glob.glob(os.path.join(log_dir, "mcts_acc_*.log"))
    
    mcts_pattern = re.compile(rf"FINAL simulation: {target_sim}, mcts accuracy: ([\d.]+)%")
    ssa_pattern = re.compile(r"ssa_([\d\.-]+) accuracy: ([\d.]+)%")

    for path in file_paths:
        filename = os.path.basename(path)
        elo_match = re.search(r'mcts_acc_(\d+)_(\d+)', filename)
        if not elo_match: continue
        
        min_elo = int(elo_match.group(1))
        max_elo = int(elo_match.group(2))
        rank_label = f"Rank {(min_elo - 1500) // 400} ({min_elo}-{max_elo})"

        with open(path, 'r', encoding='utf-8') as f:
            target_line = ""
            for line in f:
                if f"FINAL simulation: {target_sim}," in line:
                    target_line = line
                    break
            
            if not target_line: continue
            
            mcts_match = mcts_pattern.search(target_line)
            if not mcts_match: continue
            
            mcts_acc = float(mcts_match.group(1))
            row = {'Rank': rank_label, 'SE-MCTS': mcts_acc}
            
            ssa_matches = ssa_pattern.findall(target_line)
            for param, acc in ssa_matches:
                row[f'SSA_{param}'] = float(acc)
            
            all_data.append(row)

    if not all_data:
        print(f"No data found for simulation count: {target_sim}")
        return None

    # DataFrame化とソート
    df = pd.DataFrame(all_data)
    df['sort_key'] = df['Rank'].apply(lambda x: int(re.search(r'Rank (\d)', x).group(1)))
    df = df.sort_values('sort_key').drop('sort_key', axis=1).set_index('Rank')
    
    # 列の並び順を整理（SE-MCTSを先頭に、SSAを数値順に）
    cols = ['SE-MCTS'] + sorted([c for c in df.columns if 'SSA_' in c], 
                                key=lambda x: float(x.replace('SSA_', '')))
    df = df[cols]

    # ヒートマップの作成
    plt.figure(figsize=(14, 8))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Accuracy (%)'})
    plt.title(f'Prediction Accuracy Comparison (Simulation: {target_sim})', fontsize=16)
    plt.xlabel('Method / SSA Parameter (k)', fontsize=12)
    plt.ylabel('Rank (Elo Range)', fontsize=12)
    plt.tight_layout()
    
    # 画像とCSVの保存
    plt.savefig(f"accuracy_heatmap_sim{target_sim}.png")
    df.to_csv(f"rank_comparison_sim{target_sim}.csv")
    
    return df

# --- 実行 ---
log_dir = "./mcts_acc_log"
# 好きなシミュレーション回数（1, 100, 200, ..., 800）を指定してください
result_df = generate_rank_accuracy_report(log_dir, target_sim=800)

if result_df is not None:
    print(result_df)