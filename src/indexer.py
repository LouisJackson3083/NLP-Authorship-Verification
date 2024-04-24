from typing import List
from util import write_json, read_json


POS_PRE = {"<PAD>": 0, "<UNK>": 1}
"""Default pre-defined value-to-index mapping for POS."""


class Indexer():
    """
    A class for creating and manipulating indexing schemes that map values to indices and vice versa.

    Attributes:
        values (list): A list of unique values to be indexed.
        v2i (dict): property holding the value-to-index mapping
        i2v (dict): property holding the index-to-value mapping
        _v2i (list): A private list holding the value-to-index mapping and a boolean indicating if it's finalized.
        _i2v (list): A private list holding the index-to-value mapping and a boolean indicating if it's finalized.
    """

    def __init__(self, values: list, pre: dict = None):
        """
        Initializes the Indexer with a list of values and an optional pre-defined mapping.

        Parameters:
            values (list): The list of values to be indexed.
            pre (dict, optional): An optional dictionary for pre-defined value-to-index mappings.
        """
        pre = pre or {}
        erp = {v: k for k, v in pre.items()}

        self.values = list(set(values))
        self._v2i = [pre, False]
        self._i2v = [erp, False]

    @property
    def v2i(self) -> dict:
        """
        Lazily generates and returns the value-to-index mapping as a dictionary.
        """
        if self._v2i[1]:
            return self._v2i[0]

        for v in self.values:
            self._v2i[0][v] = len(self._v2i[0])

        self._v2i[1] = True
        return self._v2i[0]

    @property
    def i2v(self) -> dict:
        """
        Lazily generates and returns the index-to-value mapping as a dictionary.
        """
        if self._i2v[1]:
            return self._i2v[0]

        for v in self.values:
            self._i2v[0][len(self._i2v[0])] = v

        self._i2v[1] = True
        return self._i2v[0]

    def apply_v2i(self, values: List[list]):
        """
        Applies the value-to-index mapping to a list of lists of values, converting them to their corresponding indices.
        """
        return [[self.v2i[v] for v in row] for row in values]

    def apply_i2v(self, indexes: List[list]):
        """
        Applies the index-to-value mapping to a list of lists of indices, converting them back to their original values.
        """
        return [[self.i2v[i] for i in row] for row in indexes]

    def save(self, v2i_path: str, i2v_path: str) -> None:
        """
        Saves the value-to-index and index-to-value mappings to the specified file paths in JSON format.

        Parameters:
            v2i_path (str): The file path to save the value-to-index mapping.
            i2v_path (str): The file path to save the index-to-value mapping.
        """
        write_json(data=self.v2i, path=v2i_path)
        write_json(data=self.i2v, path=i2v_path)

    @staticmethod
    def load(v2i_path: str, i2v_path: str):
        indexer = Indexer([])
        indexer._v2i = [read_json(v2i_path), True]
        indexer._i2v = [read_json(i2v_path), True]
        return indexer
