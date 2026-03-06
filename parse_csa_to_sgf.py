import os
import re
import random
from tqdm import tqdm
import shutil

# ---------------------------------------------------------
# 設定値
# ---------------------------------------------------------
min_elo = 1500
interval = 400
max_elo = 4700
TARGET_PER_RANK = 900  # 各ランクの目標対局数

output_txt_root = "rank_testing_txt"

def fix_seed(seed=42):
    random.seed(seed)

fix_seed(42)

def get_rank_label(elo):
    if elo < min_elo or elo >= max_elo: return None
    r_idx = int((elo - min_elo) // interval)
    lo = min_elo + interval * r_idx
    hi = min_elo + interval * (r_idx + 1)
    return f"{lo}_{hi}"

def convert_to_minizero_line(csa_lines, black_elo, white_elo):
    segments = []
    segments.append(f"'black_rate:dummy:{black_elo}.0")
    segments.append(f"'white_rate:dummy:{white_elo}.0")
    for line in csa_lines:
        line = line.strip()
        if not line: continue
        if line.startswith(('P', '+', '-', '%')):
            segments.append(line)
    return ",".join(segments)

def process_to_rank_txt(csa_dir, files):
    if os.path.exists(output_txt_root):
        shutil.rmtree(output_txt_root)
    os.makedirs(output_txt_root, exist_ok=True)

    success_counts = {}
    # ランクのラベル一覧を事前に計算（終了判定用）
    all_ranks = []
    for elo in range(min_elo, max_elo, interval):
        all_ranks.append(get_rank_label(elo))

    pbar = tqdm(files, desc="Processing CSA to Ranked TXT")
    for filename in pbar:
        # すべてのランクが目標に達しているかチェック
        if all(success_counts.get(r, 0) >= TARGET_PER_RANK for r in all_ranks):
            print("\nAll ranks reached the target move count. Finishing early.")
            break

        file_path = os.path.join(csa_dir, filename)
        black_rate, white_rate = None, None
        csa_relevant_lines = []
        
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line.startswith("'black_rate:"):
                    m = re.search(r"'black_rate:[^:]+:(\d+)", line)
                    if m: black_rate = int(m.group(1))
                elif line.startswith("'white_rate:"):
                    m = re.search(r"'white_rate:[^:]+:(\d+)", line)
                    if m: white_rate = int(m.group(1))
                if line.startswith(('P', '+', '-', '%')):
                    csa_relevant_lines.append(line)
            
            if black_rate is None or not csa_relevant_lines:
                continue

            label = get_rank_label(black_rate)
            if label is None:
                continue

            # すでにこのランクが目標数に達していたらスキップ
            if success_counts.get(label, 0) >= TARGET_PER_RANK:
                continue

            single_line_game = convert_to_minizero_line(csa_relevant_lines, black_rate, white_rate)
            
            txt_file_path = os.path.join(output_txt_root, f"{label}.txt")
            with open(txt_file_path, "a", encoding="utf-8") as out_f:
                out_f.write(single_line_game + "\n")
            
            success_counts[label] = success_counts.get(label, 0) + 1
            
            # 進捗バーに現在の収集状況を表示
            pbar.set_postfix({"min_rank_count": min(success_counts.values()) if success_counts else 0})

        except Exception:
            continue
            
    return success_counts

def main():
    csa_dir = "/media/katiogoto/abb1e836-e529-4059-80be-167839ba757a/wdoor2025/2025"
    if not os.path.exists(csa_dir):
        print(f"Directory not found: {csa_dir}")
        return

    files = [f for f in os.listdir(csa_dir) if f.endswith(".csa")]
    # シャッフルして偏りを防ぐ
    random.shuffle(files)
    
    # 目標数に達するまで処理するため、入力ファイル制限を外す（または大きくする）
    counts = process_to_rank_txt(csa_dir, files)
    
    print(f"\nProcessing complete. Files saved in '{output_txt_root}/'")
    for label, count in sorted(counts.items()):
        print(f"  {label}.txt: {count} games")

if __name__ == "__main__":
    main()