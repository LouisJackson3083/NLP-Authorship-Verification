import os
import numpy as np
import logging
import itertools
import transformers
from transformers import T5Tokenizer, AutoTokenizer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
import torch


from config import DefaultConfig
from model_data.data_module import DataModule
from indexer import Indexer
import prepare_data as pd
from models.util import build_checkpoint_callback
from models.t5 import Classifier
from util import write_json


if __name__ == "__main__":
    # ............. Create Config ..............
    CONFIG = DefaultConfig().parse()

    # ............... Load Data ................
    TEST = pd.prepare_data(os.path.join(CONFIG.data_dir, CONFIG.data_test), ratio=CONFIG.ratio, labels=False)

    # ............... Tokenizer ................
    TOKENIZER = T5Tokenizer.from_pretrained(CONFIG.t5_language_model_path)
        
    # ............... Load Model ...............
    MODEL = Classifier.load_from_checkpoint(
        "/home/aaron/Programming/uni/NLU/NLP-Authorship-Verification/assets/saved/AV/version_8/checkpoints/QTag-epoch=01-val_acc=0.57.ckpt"
    )

    # .......... Feature Extraction ............
    PREP_TEST = pd.PrepDataset(TEST)
    PREP_TEST.ExtractFeatures()

    pos_indexer = Indexer.load(f"{CONFIG.saved_models_dir}v2i.json", f"{CONFIG.saved_models_dir}i2v.json")
    PREP_TEST.index_data(pos_indexer, None)

    # .......... Create Data Module ............

    DATA = {
        "test": PREP_TEST,
    }

    DATA_MOD = DataModule(data=DATA, tokenizer=TOKENIZER, config=CONFIG)
    DATA_MOD.setup()

    PREDICTIONS = []

    for i_batch, sample_batched in enumerate(DATA_MOD.test_dataloader()):
        OUTPUT = MODEL(sample_batched)
        OUTPUT = torch.softmax(OUTPUT, dim=1)
        TARGETS = np.argmax(OUTPUT.detach().numpy(), axis=1)
        PREDICTIONS.extend(TARGETS)

    # ............. Make Prediction ............
    print(PREDICTIONS)
