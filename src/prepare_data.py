import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from typing import List, Dict
import re
import string
import emoji
from tqdm.autonotebook import tqdm
import logging

# Ours
from indexer import Indexer


logger = logging.getLogger(__name__)


def split_data(
        full: Dict[str, list],
        split: float = 0.1
        ) -> [List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    Splits a set of first and second texts and their labels into a training/test split
    args:
        FIRST_TEXTS,
        SECOND_TEXTS,
        LABELS,
        test_split: float - the amount of test data points to extract from the existing set of data
    returns:
        2 sets of data
    """
    test, train = {}, {}

    train["fst_texts"], test["fst_texts"], train["snd_texts"], test["snd_texts"] = \
        train_test_split(full["fst_texts"], full["snd_texts"],
                         test_size=split, random_state=6969)

    if "labels" in full:
        train["labels"], test["labels"] = \
            train_test_split(full["labels"], test_size=split, random_state=6969)

    return train, test


def prepare_data(csv_filepath: str) -> Dict[str, list]:
    """
    Prepares the data from csv files
    args:
        csv_filepath: str - the filepath of the csv file to load our authorship verification data from
        verbose: bool
    returns:
        FIRST_TEXTS: List[str] - list of strings in the first column of csv
        SECOND_TEXTS: List[str] - list of strings in the second column of csv
        LABELS: List[int] - one hot encoded labels for whether the first and second text are from the same author
    """
    FIRST_TEXTS, SECOND_TEXTS, LABELS = [], [], []

    # Load the dataframe
    df = pd.read_csv(csv_filepath)
    # Iterate through and add to our first and second texts and labels
    # I was initially worried about this being slow but it manages to do this on the train dataset very fast
    for i in range(len(df)):
        FIRST_TEXTS.append(df.iloc[i, 0])
        SECOND_TEXTS.append(df.iloc[i, 1])
        LABELS.append(df.iloc[i, 2])

    # Ensure that the data is valid
    assert len(FIRST_TEXTS) == len(SECOND_TEXTS) == len(LABELS)

    logger.info(f"Prepared {len(df)} data points.")
    return {"fst_texts": FIRST_TEXTS, "snd_texts": SECOND_TEXTS, "labels": LABELS}


class PrepDataset():
    """
    For a given dataset, whether that's the training, dev, or
    test dataset, this class handles most of it's setup and functions.

    We will do all tokenization, feature extraction within this class and it will be what we pass to the model etc.

    Also, some texts are invalid or aren't actually pairs, we collate a list of indexes to pop from first_texts once all
    the pos, punctuation and info is calculated

    Variables:
        FIRST_TEXTS - a list of the first texts,
        SECOND_TEXTS - a list of the second texts,
        LABELS - the binary classification (0 or 1) of whether this is from the same author,
        FIRST_TOKENS - a list of the first texts tokenized,
        SECOND_TOKENS - a list of the first texts tokenized,
        FIRST_POS - a list of the first texts with the POS tagged,
        SECOND_POS - a list of the second texts with the POS tagged,
        FIRST_PUNCTUATION - a list of the first texts' punctuation,
        SECOND_PUNCTUATION - a list of the second texts' punctuation,
        FIRST_INFO - a list of the first texts' auxillary info,
        SECOND_INFO - a list of the second texts' auxillary info,
        INVALID_INDEXES - a set of indexes that are invalid and should be popped at the end of feature extraction
    """

    def __init__(self, map: dict):
        self.FIRST_TEXTS = map["fst_texts"]
        self.SECOND_TEXTS = map["snd_texts"]
        self.LABELS = map["labels"]
        self.INVALID_INDEXES = []
        self.prep_state = "initial"

    def ExtractFeatures(self):
        self.FIRST_POS = self.ExtractPOS(self.FIRST_TEXTS)
        self.SECOND_POS = self.ExtractPOS(self.SECOND_TEXTS)

        self.FIRST_PUNCTUATION = self.ExtractPunctuation(self.FIRST_TEXTS)
        self.SECOND_PUNCTUATION = self.ExtractPunctuation(self.SECOND_TEXTS)

        self.FIRST_INFORMATION = self.ExtractInformation(self.FIRST_TEXTS)
        self.SECOND_INFORMATION = self.ExtractInformation(self.SECOND_TEXTS)

        self.CleanUpData()
        self.prep_state = "extracted"

    def ExtractPOS(self, TEXTS):
        POS = []
        # For each text in the list
        for index, text in tqdm(enumerate(TEXTS), total=len(TEXTS),
                                leave=False, desc="Extracting POS"):
            # For each word, get it's part of speech
            try:
                tags = pos_tag(word_tokenize(text))
                POS.append([
                    token for _word, token in tags
                ])
                assert POS[-1] is not []
            except Exception:
                self.INVALID_INDEXES.append(index)
                POS.append([])

        return POS

    def ExtractPunctuation(self, TEXTS):
        PUNCTUATION = []
        # Get sets of all punctuation and all emojis
        all_punctuation = set(string.punctuation)
        all_emojis = set(emoji.EMOJI_DATA)
        # We then append both punctuation and emojis to our symbols variable,
        # and remove the < > tag markers
        valid_symbols = all_punctuation | all_emojis
        valid_symbols.remove(">")
        valid_symbols.remove("<")

        pattern = r"(?<=\<).*?(?=\>)"
        for index, text in tqdm(enumerate(TEXTS), total=len(TEXTS),
                                leave=False, desc="Extracting Punctuation"):
            try:
                text = re.sub(pattern, "", text)
                punc = " ".join(ch for ch in text if ch in valid_symbols)
                PUNCTUATION.append(punc)
            except Exception:
                self.INVALID_INDEXES.append(index)
                PUNCTUATION.append([])

        return PUNCTUATION

    def ExtractInformation(self, TEXTS):
        INFORMATION = []
        pattern = r"(?<=\<).*?(?=\>)"

        # Iterate through every text
        for index, text in tqdm(enumerate(TEXTS), total=len(TEXTS),
                                leave=False, desc="Extracting Information"):
            try:
                text = re.findall(pattern, text)
                INFORMATION.append(" ".join(text))
            except Exception:
                self.INVALID_INDEXES.append(index)
                INFORMATION.append([])

        return INFORMATION

    def CleanUpData(self):
        try:  # Before we clean up, let's make sure we actually have all our lists.
            self.FIRST_POS
            self.SECOND_POS
            self.FIRST_PUNCTUATION
            self.SECOND_PUNCTUATION
            self.FIRST_INFORMATION
            self.SECOND_INFORMATION
        except Exception:
            print("Not every feature has been extracted. Please check your code and try again.")
            return

        # Convert list into a set to avoid duplicates
        self.INVALID_INDEXES = set(self.INVALID_INDEXES)
        self.INVALID_INDEXES = sorted(self.INVALID_INDEXES, reverse=True)

        for index in self.INVALID_INDEXES:
            self.FIRST_TEXTS.pop(index)
            self.SECOND_TEXTS.pop(index)
            self.FIRST_POS.pop(index)
            self.SECOND_POS.pop(index)
            self.FIRST_PUNCTUATION.pop(index)
            self.SECOND_PUNCTUATION.pop(index)
            self.FIRST_INFORMATION.pop(index)
            self.SECOND_INFORMATION.pop(index)

        logger.info("Cleaned up all data!")

    def index_data(self, pos_indexer: Indexer, label_indexer: Indexer = None):
        assert self.prep_state == "extracted"

        # This should be refactored it's silly AF
        if self.LABELS:
            assert label_indexer is not None
            self.LABELS_INDEXED = label_indexer.apply_v2i(
                [[label] for label in self.LABELS]
            )
        else:
            self.LABELS_INDEXED = None

        self.FIRST_POS_INDEXED = pos_indexer.apply_v2i(self.FIRST_POS)
        self.SECOND_POS_INDEXED = pos_indexer.apply_v2i(self.SECOND_POS)
        self.prep_state = "indexed"
