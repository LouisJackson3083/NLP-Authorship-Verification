import pandas as pd
from sklearn.model_selection import train_test_split

# Feature extraction
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from typing import List
import re
import string
import emoji

from indexer import Indexer

try:
    from IPython import get_ipython
    if 'IPKernelApp' not in get_ipython().config:
        raise ImportError("Not in a notebook")
    from tqdm.notebook import tqdm
except Exception:
    from tqdm import tqdm


def split_data(
        FIRST_TEXTS: List[str],
        SECOND_TEXTS: List[str],
        LABELS: List[int],
        test_split: float = 0.1
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

    # Init test lists and amended lists
    TEST_FIRST_TEXTS, TEST_SECOND_TEXTS, TEST_LABELS = [], [], []
    TRAIN_FIRST_TEXTS, TRAIN_SECOND_TEXTS, TRAIN_LABELS = [], [], []

    TRAIN_FIRST_TEXTS, TEST_FIRST_TEXTS, TRAIN_SECOND_TEXTS, \
    TEST_SECOND_TEXTS, TRAIN_LABELS, TEST_LABELS = train_test_split(
        FIRST_TEXTS,
        SECOND_TEXTS,
        LABELS,
        test_size=test_split,
        random_state=1234
    )

    return TRAIN_FIRST_TEXTS, TRAIN_SECOND_TEXTS, TRAIN_LABELS, TEST_FIRST_TEXTS, TEST_SECOND_TEXTS, TEST_LABELS


def prepare_data(csv_filepath: str, verbose: bool = False) -> [List[str], List[str], List[int]]:
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
    FIRST_TEXTS, SECOND_TEXTS, LABELS = [],[],[]

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
    if verbose:
        print("Prepared", len(df), "data points.")

    return FIRST_TEXTS, SECOND_TEXTS, LABELS


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

    def __init__(self, FIRST_TEXTS, SECOND_TEXTS, LABELS=None):
        self.FIRST_TEXTS = FIRST_TEXTS
        self.SECOND_TEXTS = SECOND_TEXTS
        self.LABELS = LABELS
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

        print("Cleaned up all data!")

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
