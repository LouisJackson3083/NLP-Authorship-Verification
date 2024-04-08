import os
import logging
import itertools
from transformers import T5Tokenizer

from config import DefaultConfig
from model_data import ModelDataset
import indexer
import prepare_data as pd

if __name__ == "__main__":
    # ............. Create Config ..............
    CONFIG = DefaultConfig().parse()
    # ............. Setup Logging ..............
    logging.basicConfig(level=CONFIG.log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # ............... Load Data ................
    TRAIN = pd.prepare_data(os.path.join(CONFIG.data_dir, CONFIG.data_train))
    DEV = pd.prepare_data(os.path.join(CONFIG.data_dir, CONFIG.data_dev))

    TRAIN, TEST = pd.split_data(TRAIN)

    logging.info(f"Training data size: {len(TRAIN['fst_texts'])}")
    logging.info(f"Dev data size: {len(DEV['fst_texts'])}")
    logging.info(f"Test data size: {len(TEST['fst_texts'])}")

    # ............... Tokenizer ................
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")

    # ......,,,, Feature Extraction ............
    # PREP_TRAIN = pd.PrepDataset(TRAIN)
    # PREP_TRAIN.ExtractFeatures()
    # PREP_DEV = pd.PrepDataset(DEV)
    # PREP_DEV.ExtractFeatures()
    PREP_TEST = pd.PrepDataset(TEST)
    PREP_TEST.ExtractFeatures()

    # change to train for this section
    label_indexer = indexer.Indexer(values=PREP_TEST.LABELS)
    POS_VALS = list(itertools.chain(*PREP_TEST.FIRST_POS + PREP_TEST.SECOND_POS))
    pos_indexer = indexer.Indexer(values=POS_VALS, pre=indexer.POS_PRE)

    # PREP_TRAIN.index_data(pos_indexer, label_indexer)
    # PREP_DEV.index_data(pos_indexer, label_indexer)
    PREP_TEST.index_data(pos_indexer, label_indexer)

    TEST_SET = ModelDataset(PREP_TEST, tokenizer, CONFIG.max_len)

    print(TEST_SET[11])
