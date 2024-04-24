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

    # ............... Load Data ................
    TEST = pd.prepare_data(os.path.join(CONFIG.data_dir, CONFIG.data_test), ratio=CONFIG.ratio, labels=False)

    # ............... Tokenizer ................
    if CONFIG.mamba:
        TOKENIZER = AutoTokenizer.from_pretrained(CONFIG.mamba_language_model_path)
    else:
        TOKENIZER = T5Tokenizer.from_pretrained(CONFIG.t5_language_model_path)
        
    # ............... Load Model ...............
    MODEL = Classifier.load_from_checkpoint("C:/Users/neoni/OneDrive/Documents/UNI/NLU/NLP-Authorship-Verification/assets/saved/AV/version_0/checkpoints/QTag-epoch=02-val_acc=0.67.ckpt")

    # .......... Feature Extraction ............
    PREP_TEST = pd.PrepDataset(TEST)
    PREP_TEST.ExtractFeatures()

    POS_VALS = list(itertools.chain(*PREP_TEST.FIRST_POS +PREP_TEST.SECOND_POS))
    pos_indexer = indexer.Indexer(values=POS_VALS, pre=indexer.POS_PRE)

    PREP_TEST.index_data(pos_indexer, None)

    # .......... Create Data Module ............
    
    DATASET = pd.AVPREPAREDATASET(
        map=PREP_TEST,
        tokenizer=TOKENIZER,
        max_len=CONFIG.max_len
    )

    DATALOADER = torch.utils.data.DataLoader(
        DATASET, 
        batch_size=ARGS.batch_size,
        shuffle=False,
        num_workers=ARGS.num_workers
    )

    # ............. Make Prediction ............
    PREDICTIONS = []

    # for i_batch, sample_batched in enumerate(DATALOADER):
    #     OUTPUT = MODEL(sample_batched)
    #     OUTPUT = torch.softmax(OUTPUT, dim=1)
    #     TARGETS = np.argmax(OUTPUT.detach().numpy(), axis=1)
    #     PREDICTIONS.extend(TARGETS)
    # print(PREDICTIONS)