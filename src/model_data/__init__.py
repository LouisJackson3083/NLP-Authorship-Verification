import torch
import transformers

from ..prepare_data import PrepDataset


class ModelDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data: PrepDataset,
                 tokenizer: transformers.AutoTokenizer.from_pretrained,
                 max_len: int):
        assert data.prep_state == "indexed"

        self.fst_texts = data.FIRST_TEXTS
        self.snd_texts = data.SECOND_TEXTS

        self.fst_puncs = data.FIRST_PUNCTUATION
        self.snd_puncs = data.SECOND_PUNCTUATION

        self.fst_infos = data.FIRST_INFORMATION
        self.snd_infos = data.SECOND_INFORMATION

        self.fst_iposs = data.FIRST_POS_INDEXED
        self.snd_iposs = data.SECOND_POS_INDEXED

        self.labels = data.LABELS_INDEXED

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.fst_texts)

    def _pair_tokenizer(self, fst_text: str, snd_text: str):
        batch = self.tokenizer.encode_plus(
            text=fst_text,
            text_pair=snd_text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_tensors="pt",
            padding="max_length",
            truncation="longest_first",
            return_token_type_ids=True
        )

        return batch

    def __getitem__(self, index):
        fst_text = self.fst_texts[index]
        snd_text = self.snd_texts[index]
        fst_ipos = self.fst_iposs[index]
        snd_ipos = self.snd_iposs[index]
        fst_punc = self.fst_puncs[index]
        snd_punc = self.snd_puncs[index]
        fst_info = self.fst_infos[index]
        snd_info = self.snd_infos[index]

        text = self._pair_tokenizer(fst_text, snd_text)
        ipos = self._pair_tokenizer(fst_ipos, snd_ipos)
        punc = self._pair_tokenizer(fst_punc, snd_punc)
        info = self._pair_tokenizer(fst_info, snd_info)

        inid = text.input_ids.flatten()
        ipos = ipos.input_ids.flatten()
        punc = punc.input_ids.flatten()
        info = info.input_ids.flatten()

        ret = {
            "text": inid,
            "punc": punc,
            "info": info,
            "ipos": ipos,
        }

        if self.labels:
            label = self.labels[index]
            ret["labels"] = torch.tensor(label)

        return ret
