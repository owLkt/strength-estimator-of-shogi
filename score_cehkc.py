import os
import numpy as np

final_data_root = "/media/katiogoto/abb1e836-e529-4059-80be-167839ba757a1/shogi_training_data"
cand_dir_root = os.path.join(final_data_root, "candidate_shogi")

def check_estimates():
    if not os.path.exists(cand_dir_root):
        print(f"[Error] ディレクトリが見つかりません: {cand_dir_root}")
        return

    print(f"{'Elo Range':<15} | {'Candidate (Actual)':<20} | {'Training (Estimated)':<20}")
    print("-" * 65)

    subdirs = sorted([d for d in os.listdir(cand_dir_root) if os.path.isdir(os.path.join(cand_dir_root, d))])

    total_cand = 0
    for subdir in subdirs:
        feat_path = os.path.join(cand_dir_root, subdir, "features.npy")
        if os.path.exists(feat_path):
            try:
                data = np.load(feat_path, mmap_mode='r')
                count = data.shape[0]
                # RATIO_TRAIN(0.89) / RATIO_CAND(0.01) = 89倍
                estimate = count * 89 
                
                print(f"{subdir:<15} | {count:>18,d} | {estimate:>19,d}")
                total_cand += count
            except Exception as e:
                print(f"{subdir:<15} | [Error] 読み込み失敗: {e}")

    print("-" * 65)
    # ここも 89倍 に修正
    print(f"{'TOTAL':<15} | {total_cand:>18,d} | {total_cand * 89:>19,d}") 

if __name__ == "__main__":
    check_estimates()