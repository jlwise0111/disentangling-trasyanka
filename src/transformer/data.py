"""
Input pipeline for the transformer models (XLM-R, SlavicBERT): Cyrillic words ->
subword pieces -> padded batches, with each word's label on its first piece.

The tokenizer splits words into pieces, e.g. for XLM-R
    words:     нічога  мне  гаварыў
    pieces:    <s>  ▁нічога  ▁мне  ▁  гава  рыў  </s>
    word_ids:  None    0       1    2   2     2   None
Only the first piece of each word carries the word's label; every other position gets
PAD_LABEL (-100), which the loss skips. At prediction time, the label of each word is read
from its first piece, so every word gets exactly one label, as for the CRF and BiLSTM.

Words are lowercased: the Trasyanka transcripts have no capitals and the treebanks do,
so case would be a cue that exists in training but not on the test families.

Two spellings (SPELLINGS):
    original  the "cyrillic" column: treebank words in their own spelling (Russian они, Belarusian
              яны), Trasyanka converted from the Latin transcripts with Belarusian letters for every
              word (ani -> ані), since the language of a word can't be used to choose its letters.
    shared    every source converted from its Latin "tokens" column the same way, so a Russian
              treebank word looks like the same word in the transcripts (они -> ani -> ані).
              Removes the treebank-only letter cues (и щ ъ = RU, і ў = BE) that point the wrong
              way on Trasyanka. For Trasyanka records it gives exactly the "cyrillic" column.
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.model.data import LABEL_TO_ID, ID_TO_LABEL, PAD_LABEL
from src.utils.translit import latin_to_cyrillic

MODEL_NAMES = {
    "xlmr": "FacebookAI/xlm-roberta-base",                    # 100 languages, Belarusian included
    "slavicbert": "DeepPavlov/bert-base-bg-cs-pl-ru-cased",   # Bulgarian, Czech, Polish, Russian; no Belarusian
}
SPELLINGS = ("original", "shared")
SOURCE_FIELD = {"original": "cyrillic", "shared": "tokens"}   # the record field each spelling starts from
MAX_PIECES = 512        # the models' limit; the longest sentence in any set is 351 pieces, so nothing is cut


def load_tokenizer(model_key):
    """The fast tokenizer of MODEL_NAMES[model_key] (downloaded once, then cached)."""
    return AutoTokenizer.from_pretrained(MODEL_NAMES[model_key])


def input_words(record, spelling="original"):
    """
    The Cyrillic words a transformer reads for one record (a new list; the record is not changed).
        original  record["cyrillic"] as it is
        shared    latin_to_cyrillic of every word in record["tokens"] (the Latin column)
        input_words({"tokens": ["ani", "hovorit"], "cyrillic": ["они", "говорит"], ...}, "shared")
        -> ["ані", "говоріт"]
    Reads only the field the spelling needs: predict_words builds records with just that one field.
    Raises ValueError for any other spelling.
    """
    if spelling == "original":
        return list(record["cyrillic"])         # a copy, so changing the result never changes the record
    elif spelling == "shared":                  # the Latin tokens are already lowercase
        return [latin_to_cyrillic(token) for token in record["tokens"]]
    else:
        raise ValueError(f"Unknown spelling: {spelling}")


def align_labels(word_ids, labels):
    """
    Piece-level label ids for one sentence.
        word_ids  one entry per piece: the index of the word it belongs to, None for <s>, </s>, [CLS] ...
        labels    the word labels ("BE", "RU", ...), one per word
    The first piece of each word gets LABEL_TO_ID[its word's label]; special pieces and
    the later pieces of a word get PAD_LABEL.
        align_labels([None, 0, 1, 2, 2, 2, None], ["BE", "SHARED", "BE"])
        -> [-100, 0, 2, 0, -100, -100, -100]        (with BE=0, SHARED=2)
    """
    piece_labels = []
    previous_word_id = None
    for word_id in word_ids:
        # First piece of a word: not a special piece, and not a continuation of the previous word
        if word_id is not None and word_id != previous_word_id:
            piece_labels.append(LABEL_TO_ID[labels[word_id]])
        else:
            piece_labels.append(PAD_LABEL)
        previous_word_id = word_id
    return piece_labels


def encode_sentence(words, labels, tokenizer):
    """Tokenizes one sentence: {"input_ids": [...], "labels": [...], "word_ids": [...]} (lists, unpadded)."""
    encoding = tokenizer([word.lower() for word in words], is_split_into_words=True,
                         truncation=True, max_length=MAX_PIECES)
    word_ids = encoding.word_ids()
    return {"input_ids": encoding["input_ids"], "labels": align_labels(word_ids, labels), "word_ids": word_ids}


class SentenceDataset(Dataset):
    """Records -> encoded sentences, tokenized once up front (fast: the tokenizer is written in Rust)."""

    def __init__(self, records, tokenizer, spelling="original"):
        self.items = [encode_sentence(input_words(record, spelling), record["labels"], tokenizer)
                      for record in records]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate_batch(items, pad_id):
    """
    Pads a list of encoded sentences to the longest one in the batch. Returns
        input_ids       LongTensor [batch, pieces], padded with pad_id
        attention_mask  LongTensor [batch, pieces], 1 for real pieces, 0 for padding
        labels          LongTensor [batch, pieces], padded with PAD_LABEL
        word_ids        list of the unpadded word_ids lists (needed to read word predictions back)
    """
    longest = max(len(item["input_ids"]) for item in items)
    input_ids, attention_mask, labels, word_ids = [], [], [], []
    for item in items:
        padding = longest - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * padding)
        attention_mask.append([1] * len(item["input_ids"]) + [0] * padding)
        labels.append(item["labels"] + [PAD_LABEL] * padding)
        word_ids.append(item["word_ids"])

    return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "word_ids": word_ids}


def word_predictions(piece_label_ids, word_ids):
    """
    Word labels for one sentence from its piece-level predictions.
        piece_label_ids  predicted label id for every piece (padding positions may follow)
        word_ids         the sentence's word_ids (unpadded)
    Returns one label string per word, read from each word's first piece:
        word_predictions([3, 0, 2, 0, 1, 1, 3], [None, 0, 1, 2, 2, 2, None]) -> ["BE", "SHARED", "BE"]
    """
    word_labels = []
    previous_word_id = None
    # zip stops at the end of word_ids, so padding positions after the sentence are skipped
    for word_id, piece_label_id in zip(word_ids, piece_label_ids):
        if word_id is not None and word_id != previous_word_id:     # same first-piece rule as align_labels
            word_labels.append(ID_TO_LABEL[piece_label_id])
        previous_word_id = word_id
    return word_labels


def make_loader(records, tokenizer, batch_size, shuffle=False, spelling="original"):
    """DataLoader of padded batches. Keep shuffle=False for prediction, so the output order matches records."""
    dataset = SentenceDataset(records, tokenizer, spelling)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=lambda items: collate_batch(items, tokenizer.pad_token_id))
