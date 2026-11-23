"""
Character-level input for the hierarchical BiLSTM.

A sentence becomes a list of words, each word a list of character ids. A batch is
padded twice: every word to the longest word in the batch, and every sentence to
the longest sentence. Padded words get the label PAD_LABEL, which the loss ignores.

    records (splits.training_sentences / evaluation_sets)
      -> build_char_vocab (training records only)
      -> SentenceDataset (encodes every sentence once)
      -> DataLoader with collate_batch
      -> chars [batch, sentence, word], labels [batch, sentence], word_mask [batch, sentence]

Run from the project root for a demo:   python src/model/data.py
"""
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils.metrics import LABELS

PAD_CHAR = "<pad>"           # id 0: fills short words and padded words
UNK_CHAR = "<unk>"           # id 1: characters not in the vocabulary
PAD_LABEL = -100             # nn.CrossEntropyLoss(ignore_index=-100) skips these positions

LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}     # BE, RU, SHARED, HYBRID -> 0..3
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}

# The training data has 124 distinct characters, most of them emoji and symbols from the
# HSE web texts. Characters seen fewer than MIN_CHAR_COUNT times map to <unk>, so the
# model also learns what to do with an unknown character.
MIN_CHAR_COUNT = 5

# 99.9% of training words have at most 18 characters (longest: 33). Longer words are cut.
MAX_WORD_LEN = 20


def build_char_vocab(records, min_count=MIN_CHAR_COUNT):
    """
    Character -> id, built from the training records' "tokens" only.
    PAD_CHAR is 0 and UNK_CHAR is 1; the kept characters follow in sorted order,
    so the same records always give the same ids.
    """
    char_counts = Counter()
    for record in records:
        for token in record["tokens"]:
            char_counts.update(token)

    kept_chars = sorted(char for char, count in char_counts.items() if count >= min_count)

    vocab = {PAD_CHAR: 0, UNK_CHAR: 1}
    for char in kept_chars:
        vocab[char] = len(vocab)        # 2, 3, 4, ...
    return vocab


def encode_word(word, char_vocab, max_len=MAX_WORD_LEN):
    """
    A word as a list of character ids, at most max_len long.
    Characters missing from char_vocab become the UNK_CHAR id.
        encode_word("mne", vocab) -> [id of m, id of n, id of e]
    """
    unk_id = char_vocab[UNK_CHAR]
    return [char_vocab.get(char, unk_id) for char in word[:max_len]]


class SentenceDataset(Dataset):
    """
    Encodes each record once, up front: item i is (list of encoded words, list of label ids).
    Records without "labels" are not supported; every processed dataset has them.
    """

    def __init__(self, records, char_vocab):
        self.items = []
        for record in records:
            words = [encode_word(word, char_vocab) for word in record["tokens"]]
            labels = [LABEL_TO_ID[label] for label in record["labels"]]
            self.items.append((words, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate_batch(batch):
    """
    Pads a list of (words, labels) pairs into tensors:
        chars      long  [batch, max sentence length, max word length], PAD id 0 where empty
        labels     long  [batch, max sentence length], PAD_LABEL where there is no word
        word_mask  bool  [batch, max sentence length], True where there is a real word
    "max" means the longest sentence / word in this batch, not in the whole dataset.
    """
    # Longest sentence (in words) and longest word (in characters) in this batch
    longest_sentence = 0
    longest_word = 0
    for words, sentence_labels in batch:
        longest_sentence = max(longest_sentence, len(words))
        longest_word = max(longest_word, max(len(word) for word in words))

    # Start from all padding, then fill in the real words
    chars = torch.zeros(len(batch), longest_sentence, longest_word, dtype=torch.long)
    labels = torch.full((len(batch), longest_sentence), PAD_LABEL, dtype=torch.long)
    word_mask = torch.zeros(len(batch), longest_sentence, dtype=torch.bool)

    for b, (words, sentence_labels) in enumerate(batch):
        for s, (word, label) in enumerate(zip(words, sentence_labels)):
            chars[b, s, :len(word)] = torch.tensor(word)
            labels[b, s] = label
            word_mask[b, s] = True
    return chars, labels, word_mask


def make_loader(records, char_vocab, batch_size=32, shuffle=False):
    """DataLoader over records. Use shuffle=True for training only, so evaluation keeps record order."""
    return DataLoader(SentenceDataset(records, char_vocab), batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collate_batch)


def main():
    """Builds the vocabulary from the mixed training set and prints one small batch."""
    from src.baseline.splits import training_sentences

    train = training_sentences("mixed")
    char_vocab = build_char_vocab(train)
    print(f"{len(char_vocab)} characters (including {PAD_CHAR} and {UNK_CHAR}):",
          "".join(ch for ch in char_vocab if ch not in (PAD_CHAR, UNK_CHAR)))

    loader = make_loader(train[-3:], char_vocab, batch_size=3)
    chars, labels, word_mask = next(iter(loader))
    print("chars", tuple(chars.shape), "labels", tuple(labels.shape), "word_mask", tuple(word_mask.shape))
    for record in train[-3:]:
        print(" ", list(zip(record["tokens"], record["labels"])))
    print(labels)


if __name__ == "__main__":
    main()
