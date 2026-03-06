#!/usr/bin/env python

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from strength.trainer.create_network import create_network
import collections


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs, flush=True)


class NpyDataLoader:
    def __init__(self, root_dir, batch_size, nn_type, py_module):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.nn_type = nn_type
        self.py = py_module
        self.data = {}
        self.sorted_keys = []
        self.dummy_templates = {} 
        self.use_dummy_rank = False 
        self.data_counts = [] 
        self.total_samples = 0
        
        # r_infinity 用
        self.r_inf_data = None
        self.r_inf_count = 0

        # --- 追加: バッファ設定 ---
        self.cache = {} # 各ランク帯のデータを保持するデック
        self.cache_fill_size = 2048 # 1回のHDDアクセスで読み込む局面数（連続領域）

        if self.nn_type == "bt":
            self.num_ranks = self.py.get_bt_num_rank_per_batch()
            self.num_pos_per_rank = self.py.get_bt_num_position_per_rank()
            self.num_bt_batch = self.py.get_bt_num_batch_size()
            derived_batch_size = self.num_bt_batch * self.num_ranks * self.num_pos_per_rank
            if self.batch_size != derived_batch_size:
                print(f"[Warning] Adjusting batch size to {derived_batch_size}")
                self.batch_size = derived_batch_size

        self.load_all_data()

        # ロード後に各ランクのバッファを初期化
        for key in self.sorted_keys:
            self.cache[key] = collections.deque()

    def load_all_data(self):
        self.data = {}
        self.sorted_keys = []
        self.data_counts = []
        self.total_samples = 0
        
        # ディレクトリ名でソート
        all_subdirs = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])

        for subdir in all_subdirs:
            path = os.path.join(self.root_dir, subdir)
            files = {
                "features": os.path.join(path, "features.npy"),
                "policy": os.path.join(path, "policy.npy"),
                "value": os.path.join(path, "value.npy"),
                "rank": os.path.join(path, "rank.npy")
            }

            if not all(os.path.exists(p) for p in files.values()):
                continue

            loaded_data = {}
            sample_count = 0
            for k, p in files.items():
                arr = np.load(p, mmap_mode='r')
                loaded_data[k] = arr
                if k == "features":
                    sample_count = arr.shape[0]

            if not self.dummy_templates:
                self.dummy_templates = {
                    f"{k}_shape": loaded_data[k].shape[1:] for k in loaded_data
                }
                self.dummy_templates.update({
                    f"{k}_dtype": loaded_data[k].dtype for k in loaded_data
                })

            self.data[subdir] = loaded_data
            self.sorted_keys.append(subdir)
            self.total_samples += sample_count
            self.data_counts.append(self.total_samples)

        if len(self.data) == 0 and self.nn_type != "bt":
             raise RuntimeError(f"[Error] {self.root_dir} に読み込めるデータがありません！")
        
        print(f"[DataLoader] {self.root_dir} から {len(self.data)} 個のランク帯をロード")

        if self.nn_type == "bt":
            if len(self.data) == self.num_ranks:
                print("[Info] BT: 設定ランク数とフォルダ数が一致。")
                self.use_dummy_rank = False
            elif len(self.data) == self.num_ranks - 1:
                # ランク数不足(-1) -> random_data をロード
                r_inf_path = os.path.abspath(os.path.join(self.root_dir, "../random_data"))
                print(f"[Info] ランク数不足(-1)を検知。r_infinity として {r_inf_path} をロードします。")
                
                if not os.path.exists(r_inf_path):
                     raise RuntimeError(f"[Error] {r_inf_path} が見つかりません。")

                self.r_inf_data = {}
                files = ["features", "policy", "value", "rank"]
                for k in files:
                    p = os.path.join(r_inf_path, f"{k}.npy")
                    if os.path.exists(p):
                        self.r_inf_data[k] = np.load(p, mmap_mode='r')
                    elif k == "rank": pass
                    else: raise RuntimeError(f"[Error] {k}.npy missing")

                self.r_inf_count = self.r_inf_data["features"].shape[0]
                self.use_dummy_rank = True
                self.sorted_keys.insert(0, "_DUMMY_RANK_")
                print(f"[Info] r_infinity データをロード完了 ({self.r_inf_count} samples)")
            else:
                raise RuntimeError(f"[Error] BTモード: 設定ランク数({self.num_ranks})と、ロードしたディレクトリ数({len(self.data)})が一致しません。")
            
    def _replenish_cache(self, key):
        """キャッシュが空になったらHDDから連続したブロックを読み込む"""
        if key == "_DUMMY_RANK_":
            n = self.r_inf_count
            source = self.r_inf_data
        else:
            n = self.data[key]["features"].shape[0]
            source = self.data[key]

        # ランダムな開始地点を決定
        start_idx = np.random.randint(0, max(1, n - self.cache_fill_size))
        end_idx = min(n, start_idx + self.cache_fill_size)

        # HDDから連続範囲を一括スライス (.copy()で実メモリへ引き込む)
        f_block = source["features"][start_idx:end_idx].copy()
        p_block = source["policy"][start_idx:end_idx].copy()
        v_block = source["value"][start_idx:end_idx].copy()
        
        # ランクIDの生成
        rank_idx = self.sorted_keys.index(key)
        rank_data = np.array([rank_idx], dtype=np.float32)

        # ブロック内で局面をシャッフル（学習の質を保つため）
        block_indices = list(range(len(f_block)))
        np.random.shuffle(block_indices)

        for i in block_indices:
            self.cache[key].append({
                "features": f_block[i],
                "policy": p_block[i],
                "value": v_block[i],
                "rank": rank_data
            })

    def _get_buffered_sample(self, key):
        """バッファから1局面取り出す。空なら補充する。"""
        if not self.cache[key]:
            self._replenish_cache(key)
        return self.cache[key].popleft()

    def sample_data(self, device="cpu"):
        features_list, policy_list, value_list, rank_list = [], [], [], []

        if self.nn_type == "bt":
            for _ in range(self.num_bt_batch):
                for i in range(self.num_ranks):
                    key = self.sorted_keys[i]
                    # バッファから指定数だけ取得
                    for _ in range(self.num_pos_per_rank):
                        d = self._get_buffered_sample(key)
                        features_list.append(d["features"])
                        policy_list.append(d["policy"])
                        value_list.append(d["value"])
                        rank_list.append(d["rank"])
        else:
            # Alphazeroモード等の全体サンプリング
            # キーをランダムに選んでバッファから取得
            for _ in range(self.batch_size):
                key = np.random.choice(self.sorted_keys)
                d = self._get_buffered_sample(key)
                features_list.append(d["features"])
                policy_list.append(d["policy"])
                value_list.append(d["value"])
                rank_list.append(d["rank"])

        # Tensor化して指定デバイスへ転送
        features = torch.tensor(np.array(features_list), dtype=torch.float32, device=device)
        policy   = torch.tensor(np.array(policy_list), dtype=torch.float32, device=device)
        value    = torch.tensor(np.array(value_list), dtype=torch.float32, device=device)
        rank     = torch.tensor(np.array(rank_list), dtype=torch.float32, device=device)

        if self.nn_type == "alphazero":
            return features, policy, value
        else:
            return features, policy, value, rank
        

class Model:
    def __init__(self):
        self.training_step = 0
        self.network = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optimizer = None
        self.scheduler = None

    def load_model(self, training_dir, model_file):
        self.training_step = 0
        self.network = create_network(py.get_game_name(),
                                      py.get_nn_num_input_channels(),
                                      py.get_nn_input_channel_height(),
                                      py.get_nn_input_channel_width(),
                                      py.get_nn_num_hidden_channels(),
                                      py.get_nn_hidden_channel_height(),
                                      py.get_nn_hidden_channel_width(),
                                      py.get_nn_num_blocks(),
                                      py.get_nn_action_size(),
                                      py.get_nn_num_value_hidden_channels(),
                                      py.get_nn_discrete_value_size(),
                                      py.get_nn_rank_size(),
                                      py.get_nn_type_name())
        self.network.to(self.device)
        self.optimizer = optim.SGD(self.network.parameters(),
                                   lr=py.get_learning_rate(),
                                   momentum=py.get_momentum(),
                                   weight_decay=py.get_weight_decay())
        step_sizes = [100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000]
        #step_sizes = [2000, 4000, 6000, 8000, 30000, 35000, 40000, 45000]
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=step_sizes, gamma=0.5)

        if model_file:
            snapshot = torch.load(f"{training_dir}/model/{model_file}", map_location=torch.device('cpu'))
            self.training_step = snapshot['training_step']
            self.network.load_state_dict(snapshot['network'])
            self.optimizer.load_state_dict(snapshot['optimizer'])
            self.optimizer.param_groups[0]["lr"] = py.get_learning_rate()
            self.scheduler.load_state_dict(snapshot['scheduler'])

        self.network = nn.DataParallel(self.network)

    def save_model(self, training_dir):
        snapshot = {'training_step': self.training_step,
                    'network': self.network.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()}
        torch.save(snapshot, f"{training_dir}/model/weight_iter_{self.training_step}.pkl")
        torch.jit.script(self.network.module).save(f"{training_dir}/model/weight_iter_{self.training_step}.pt")


def calculate_bt_loss(network_output, label_rank):
    score_reshape = network_output["score"].view(py.get_bt_num_batch_size(), py.get_bt_num_rank_per_batch(), py.get_bt_num_position_per_rank())
    if py.get_bt_use_weight():
        weight_reshape = network_output["weight"].view(py.get_bt_num_batch_size(), py.get_bt_num_rank_per_batch(), py.get_bt_num_position_per_rank())
    else:
        weight_reshape = torch.ones((py.get_bt_num_batch_size(), py.get_bt_num_rank_per_batch(), py.get_bt_num_position_per_rank())).to(network_output["score"].device)
    average_score = (score_reshape * weight_reshape).sum(dim=2) / weight_reshape.sum(dim=2)

    loss_bt = 0
    accuracy_bt = 0
    for i in range(2, py.get_bt_num_rank_per_batch() + 1):
        labels = torch.zeros(py.get_bt_num_batch_size(), i, dtype=average_score.dtype, device=average_score.device)
        labels[:, i - 1] = 1
        loss_bt += -(labels * nn.functional.log_softmax(average_score[:, :i], dim=1)).sum(dim=1).mean()
        _, max_output = torch.max(average_score[:, :i], dim=1)
        _, max_label = torch.max(labels, dim=1)
        accuracy_bt += (max_output == max_label).float().mean().item()
    loss_bt /= (py.get_bt_num_rank_per_batch() - 1)
    accuracy_bt /= (py.get_bt_num_rank_per_batch() - 1)
    return loss_bt, accuracy_bt


def add_training_info(training_info, key, value):
    if key not in training_info:
        training_info[key] = 0
    training_info[key] += value


def calculate_accuracy(output, label, batch_size):
    max_output = np.argmax(output.to('cpu').detach().numpy(), axis=1)
    max_label = np.argmax(label.to('cpu').detach().numpy(), axis=1)
    return (max_output == max_label).sum() / batch_size


def train(model, training_dir, data_loader):
    data_loader.load_all_data()

    eprint("start training ...")
    training_info = {}
    for i in range(1, py.get_training_step() + 1):
        model.optimizer.zero_grad()

        if py.get_nn_type_name() == "alphazero":
            features, label_policy, label_value = data_loader.sample_data(model.device)
            network_output = model.network(features)
            loss_policy = -(label_policy * nn.functional.log_softmax(network_output["policy_logit"], dim=1)).sum() / network_output["policy_logit"].shape[0]
            loss_value = torch.nn.functional.mse_loss(network_output["value"], label_value.unsqueeze(1))
            loss = loss_policy + loss_value

            add_training_info(training_info, 'loss_policy', loss_policy.item())
            add_training_info(training_info, 'accuracy_policy', calculate_accuracy(network_output["policy_logit"], label_policy, py.get_batch_size()))
            add_training_info(training_info, 'loss_value', loss_value.item())
        elif py.get_nn_type_name() == "rank":
            features, label_policy, label_value, label_rank = data_loader.sample_data(model.device)
            network_output = model.network(features)
            loss_policy = -(label_policy * nn.functional.log_softmax(network_output["policy_logit"], dim=1)).sum() / network_output["policy_logit"].shape[0]
            loss_value = torch.nn.functional.mse_loss(network_output["value"], label_value.unsqueeze(1))
            loss_rank = -(label_rank * nn.functional.log_softmax(network_output["rank_logit"], dim=1)).sum() / network_output["rank_logit"].shape[0]
            loss = loss_policy + loss_value + loss_rank

            add_training_info(training_info, 'loss_policy', loss_policy.item())
            add_training_info(training_info, 'accuracy_policy', calculate_accuracy(network_output["policy_logit"], label_policy, py.get_batch_size()))
            add_training_info(training_info, 'loss_value', loss_value.item())
            add_training_info(training_info, 'loss_rank', loss_rank.item())
            add_training_info(training_info, 'accuracy_rank', calculate_accuracy(network_output["rank_logit"], label_rank, py.get_batch_size()))
        
        elif py.get_nn_type_name() == "bt":
            features, label_policy, label_value, label_rank = data_loader.sample_data(model.device)
            network_output = model.network(features)
            
            # --- Mask Processing for r_infinity ---
            rank_indices = label_rank.view(-1)
            valid_mask = (rank_indices != 0).float() 
            num_valid = valid_mask.sum()

            if num_valid > 0:
                log_softmax = nn.functional.log_softmax(network_output["policy_logit"], dim=1)
                per_sample_policy_loss = -(label_policy * log_softmax).sum(dim=1)
                loss_policy = (per_sample_policy_loss * valid_mask).sum() / num_valid
            else:
                loss_policy = torch.tensor(0.0, device=model.device)

            if num_valid > 0:
                per_sample_mse = nn.functional.mse_loss(network_output["value"], label_value.unsqueeze(1), reduction='none').view(-1)
                loss_value = (per_sample_mse * valid_mask).sum() / num_valid
            else:
                loss_value = torch.tensor(0.0, device=model.device)

            loss_bt, accuracy_bt = calculate_bt_loss(network_output, label_rank)
            
            loss = loss_policy + loss_value + loss_bt

            add_training_info(training_info, 'loss_policy', loss_policy.item())
            add_training_info(training_info, 'accuracy_policy', calculate_accuracy(network_output["policy_logit"], label_policy, py.get_batch_size()))
            add_training_info(training_info, 'loss_value', loss_value.item())
            add_training_info(training_info, 'loss_bt', loss_bt.item())
            add_training_info(training_info, 'accuracy_bt', accuracy_bt)

        loss.backward()
        model.optimizer.step()
        model.scheduler.step()

        model.training_step += 1
        if model.training_step != 0 and model.training_step % py.get_training_display_step() == 0:
            if model.training_step % (5 * py.get_training_display_step()) == 0:
                model.save_model(training_dir)
            eprint("[{}] nn step {}, lr: {}.".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), model.training_step, round(model.optimizer.param_groups[0]["lr"], 6)))
            for loss in training_info:
                eprint("\t{}: {}".format(loss, round(training_info[loss] / py.get_training_display_step(), 5)))
            training_info = {}


if __name__ == '__main__':
    if len(sys.argv) == 5:
        game_type = sys.argv[1]
        training_dir = sys.argv[2]
        model_file = sys.argv[3]
        conf_file_name = sys.argv[4]

        _temps = __import__(f'build.{game_type}', globals(), locals(), ['strength_py'], 0)
        py = _temps.strength_py
    else:
        eprint("python train.py game_type training_dir model_file_name conf_file")
        exit(0)

    if py.load_config_file(conf_file_name) is False:
        eprint(f"Failed to load config file {conf_file_name}")
        exit(0)

    config_data_path = py.get_training_sgf_dir()
    
    # パスを絶対パスに変換して確実に認識させる
    root_data_dir = os.path.abspath(config_data_path)
    print(f"[Info] Loading data from: {root_data_dir}")

    data_loader = NpyDataLoader(
        root_dir=root_data_dir, 
        batch_size=py.get_batch_size(),
        nn_type=py.get_nn_type_name(),
        py_module=py)
    model = Model()

    model_file = model_file.replace('"', '')
    model.load_model(training_dir, model_file)
    if model_file:
        train(model, training_dir, data_loader)
    else:
        model.save_model(training_dir)