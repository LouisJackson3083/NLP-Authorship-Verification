import pytorch_lightning as pl
import argparse
import transformers
import torch

from . import ModelDataset


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 data: dict,
                 tokenizer: transformers.AutoTokenizer.from_pretrained,
                 config: argparse.ArgumentParser.parse_args):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.data = data
        self.model_data = {}

    def setup(self, stage=None):
        self.model_data = {
            k: ModelDataset(dataset, self.tokenizer, self.config.max_len)
            for k, dataset in self.data.items()
        }

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.model_data["train"],
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.model_data["val"],
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.model_data["test"],
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )
