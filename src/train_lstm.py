import os
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
import indexer
import prepare_data as pd
from models.util import build_checkpoint_callback
from models.lstmcrook import CrookClassifier
from util import write_json

if __name__ == "__main__":
    # ............. Create Config ..............
    CONFIG = DefaultConfig().parse()

    # could change
    torch.set_float32_matmul_precision('high')

    # ............. Setup Logging ..............
    logging.basicConfig(level=CONFIG.log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    LOGGER = CSVLogger(save_dir=CONFIG.saved_models_dir, name=CONFIG.model_name)
    transformers.logging.set_verbosity_error()

    # ............... Load Data ................
    TRAIN = pd.prepare_data(os.path.join(CONFIG.data_dir, CONFIG.data_train), ratio=CONFIG.ratio)
    DEV = pd.prepare_data(os.path.join(CONFIG.data_dir, CONFIG.data_dev), ratio=CONFIG.ratio)

    TRAIN, TEST = pd.split_data(TRAIN)

    logging.info(f"Training data size: {len(TRAIN['fst_texts'])}")
    logging.info(f"Dev data size: {len(DEV['fst_texts'])}")
    logging.info(f"Test data size: {len(TEST['fst_texts'])}")

    # ............... Tokenizer ................
    if CONFIG.model2:
        # TOKENIZER = T5Tokenizer.from_pretrained(CONFIG.t5_language_model_path)
        TOKENIZER = AutoTokenizer.from_pretrained(CONFIG.model2_language_model_path)
    else:
        TOKENIZER = T5Tokenizer.from_pretrained(CONFIG.t5_language_model_path)

    # .......... Feature Extraction ............
    PREP_TRAIN = pd.PrepDataset(TRAIN)
    PREP_TRAIN.ExtractFeatures()
    PREP_DEV = pd.PrepDataset(DEV)
    PREP_DEV.ExtractFeatures()
    PREP_TEST = pd.PrepDataset(TEST)
    PREP_TEST.ExtractFeatures()

    label_indexer = indexer.Indexer(values=PREP_TRAIN.LABELS)
    POS_VALS = list(itertools.chain(*PREP_TRAIN.FIRST_POS + PREP_TRAIN.SECOND_POS +
                                    PREP_TEST.FIRST_POS + PREP_TEST.SECOND_POS +
                                    PREP_DEV.FIRST_POS + PREP_DEV.SECOND_POS))
    pos_indexer = indexer.Indexer(values=POS_VALS, pre=indexer.POS_PRE)

    pos_indexer.save(f"{CONFIG.saved_models_dir}v2i.json", f"{CONFIG.saved_models_dir}i2v.json")

    PREP_TRAIN.index_data(pos_indexer, label_indexer)
    PREP_DEV.index_data(pos_indexer, label_indexer)
    PREP_TEST.index_data(pos_indexer, label_indexer)

    print("TRAIN:",PREP_TRAIN)