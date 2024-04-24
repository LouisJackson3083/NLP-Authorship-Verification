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
from tqdm.autonotebook import tqdm
from pandas import DataFrame


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

    torch.set_float32_matmul_precision('high')
    transformers.logging.set_verbosity_error()

    # ............... Load Data ................
    TEST = pd.prepare_data(os.path.join(CONFIG.data_dir, CONFIG.data_test), labels=False, ratio=0.01)

    # ............... Tokenizer ................
    TOKENIZER = T5Tokenizer.from_pretrained(CONFIG.t5_language_model_path)
        
    # ............... Load Model ...............
    MODEL = Classifier.load_from_checkpoint(
        "./assets/saved/AV/best/best.ckpt"
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

    for i_batch, sample_batched in tqdm(enumerate(DATA_MOD.test_dataloader()), total=len(DATA_MOD.test_dataloader())):
        OUTPUT = MODEL(sample_batched)
        OUTPUT = torch.softmax(OUTPUT, dim=1)
        TARGETS = np.argmax(OUTPUT.detach().numpy(), axis=1)
        PREDICTIONS.extend(TARGETS)

    # ............. Make Prediction ............
    print(f"Predicted {len(PREDICTIONS)} samples")

    df = DataFrame(PREDICTIONS, columns=['prediction'])

    df.to_csv('predictions_Group63_C.csv', index=False)
