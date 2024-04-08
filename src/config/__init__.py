import argparse
import logging
from pathlib import Path


class DefaultConfig:
    def __init__(self):
        self.p = argparse.ArgumentParser()
        self.p.add_argument("--model_name", type=str, default="AV")
        self.p.add_argument("--save_top_k", type=int, default=1)
        self.p.add_argument("--num_workers", type=int, default=8)
        self.p.add_argument("--num_epochs", type=int, default=5)
        self.p.add_argument("--batch_size", type=int, default=4)
        self.p.add_argument("--max_len", type=int, default=350)
        self.p.add_argument("--lr", type=float, default=2e-5)
        self.p.add_argument("--num_filters", type=int, default=128)
        self.p.add_argument("--filter_sizes", type=int, default=[1, 2, 3])
        self.p.add_argument("--dropout", type=float, default=0.15)
        self.p.add_argument("--embedding_dim", type=float, default=50)

        root = Path(__file__).parents[2].__str__()
        self.p.add_argument("--data_dir", type=str, default=f"{root}/data/")
        self.p.add_argument("--data_train", type=str, default="train.csv")
        self.p.add_argument("--data_dev", type=str, default="dev.csv")
        self.p.add_argument("--asset_dir", type=str, default=f"{root}/assets/")
        self.p.add_argument("--saved_models_dir", type=str, default=f"{root}/assets/saved/")
        self.p.add_argument("--log_level", default=logging.INFO)
        self.p.add_argument("--best_model_path_file", type=str, default=f"{root}/assets/saved/av/best_model_path.json")
        self.p.add_argument("--language_model_path", type=str, default="google-t5/t5-small")

    def parse(self):
        return self.p.parse_args()
