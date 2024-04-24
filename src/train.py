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
from models.t5 import Classifier
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
                                    PREP_TEST.FIRST_POS + PREP_TEST.SECOND_POS))
    pos_indexer = indexer.Indexer(values=POS_VALS, pre=indexer.POS_PRE)

    pos_indexer.save(f"{CONFIG.saved_models_dir}v2i.json", f"{CONFIG.saved_models_dir}i2v.json")

    PREP_TRAIN.index_data(pos_indexer, label_indexer)
    PREP_DEV.index_data(pos_indexer, label_indexer)
    PREP_TEST.index_data(pos_indexer, label_indexer)

    # .......... Create Data Module ............
    DATA = {
        "train": PREP_TRAIN,
        "val": PREP_DEV,
        "test": PREP_TEST,
    }
    DATA_MOD = DataModule(data=DATA, tokenizer=TOKENIZER, config=CONFIG)
    DATA_MOD.setup()

    # ......... Create Model Trainer ...........
    EARLY_STOPPING_CALLBACK = EarlyStopping(monitor="val_acc", patience=CONFIG.patience, mode="max")
    CHECKPOINT_CALLBACK = build_checkpoint_callback(save_top_k=CONFIG.save_top_k,
                                                    monitor="val_acc",
                                                    mode="max",
                                                    filename="QTag-{epoch:02d}-{val_acc:.2f}")
    TRAINER = pl.Trainer(max_epochs=CONFIG.num_epochs, accelerator="auto",
                         callbacks=[CHECKPOINT_CALLBACK, EARLY_STOPPING_CALLBACK],
                         logger=LOGGER)

    # ............. Create Model ...............
    MODEL = Classifier(num_classes=len(label_indexer.v2i),
                       lr=CONFIG.lr,
                       max_len=CONFIG.max_len, filter_sizes=CONFIG.filter_sizes,
                       n_filters=CONFIG.num_filters,
                       config=CONFIG)

    # .........Train and Test Model ............
    TRAINER.fit(MODEL, datamodule=DATA_MOD)
    TRAINER.test(ckpt_path="best", datamodule=DATA_MOD)

    # .... save best mt5_model_en path .........
    write_json(path=CONFIG.best_model_path_file,
               data={"best_model_path": CHECKPOINT_CALLBACK.best_model_path})
