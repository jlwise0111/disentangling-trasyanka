"""
Hierarchical BiLSTM for word-level language identification.

    chars [batch, sentence, word]
      -> character embeddings                     [batch*sentence, word, CHAR_EMB_DIM]
      -> character BiLSTM, last state each way    [batch*sentence, 2*CHAR_HIDDEN]   one vector per word
      -> back to sentences                        [batch, sentence, 2*CHAR_HIDDEN]
      -> word BiLSTM over the sentence            [batch, sentence, 2*WORD_HIDDEN]  each word sees its neighbors
      -> dropout, linear layer                    [batch, sentence, NUM_LABELS]     one score per label

The output is raw scores (logits); nn.CrossEntropyLoss applies the softmax itself.
Character-only for now: no word embeddings, so it runs on the Latin "tokens" directly.

Run from the project root for a demo:   python src/model/bilstm.py
"""
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.model.data import LABEL_TO_ID

CHAR_EMB_DIM = 32       # size of one character's vector
CHAR_HIDDEN = 64        # character BiLSTM, per direction -> a word is 2 * 64 = 128 numbers
WORD_HIDDEN = 128       # word BiLSTM, per direction -> each word's output is 2 * 128 = 256 numbers
DROPOUT = 0.3
NUM_LABELS = len(LABEL_TO_ID)    # BE, RU, SHARED, HYBRID


class HierarchicalBiLSTM(nn.Module):
    """Character BiLSTM per word, word BiLSTM per sentence, linear layer per word."""

    def __init__(self, n_chars, char_emb_dim=CHAR_EMB_DIM, char_hidden=CHAR_HIDDEN,
                 word_hidden=WORD_HIDDEN, dropout=DROPOUT, num_labels=NUM_LABELS):
        super().__init__()
        # padding_idx=0 keeps the <pad> vector at zero and never trains it
        self.char_embedding = nn.Embedding(n_chars, char_emb_dim, padding_idx=0)
        self.char_lstm = nn.LSTM(char_emb_dim, char_hidden, batch_first=True, bidirectional=True)
        # A bidirectional LSTM outputs 2 * hidden numbers, so each layer takes twice the one before
        self.word_lstm = nn.LSTM(char_hidden * 2, word_hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(word_hidden * 2, num_labels)

    def encode_words(self, chars):
        """
        One vector per word from its characters.
        chars: long [batch, sentence, word]  ->  float [batch, sentence, 2 * char_hidden]
        """
        # One row per word: [batch * sentence, word]
        batch, sentence, word = chars.shape
        flat = chars.reshape(batch * sentence, word)
        embedded = self.char_embedding(flat)

        # Real characters per word. Padded word slots have 0, which pack_padded_sequence
        # refuses, so they count as 1; their junk vectors are hidden by word_mask later.
        lengths = (flat != 0).sum(dim=1).clamp(min=1)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # h_n [2, batch * sentence, char_hidden]: h_n[0] is the forward state after the last
        # real character, h_n[1] the backward state after the first one
        _, (h_n, _) = self.char_lstm(packed)
        word_vectors = torch.cat([h_n[0], h_n[1]], dim=-1)

        return word_vectors.reshape(batch, sentence, -1)

    def forward(self, chars, word_mask):
        """
        Label scores for every word position.
        chars: long [batch, sentence, word], word_mask: bool [batch, sentence]
        ->  float [batch, sentence, num_labels]  (padded positions hold junk; ignore them)
        """
        word_vectors = self.encode_words(chars)

        # Pack by sentence length (every real sentence has at least one word, so no clamp)
        sentence_lengths = word_mask.sum(dim=1)
        packed = pack_padded_sequence(word_vectors, sentence_lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)

        # The output at every word; the final states (h_n, c_n) are not needed here.
        # total_length keeps the sentence dimension equal to the labels' dimension.
        packed_output, _ = self.word_lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True,
                                        total_length=chars.shape[1])

        output = self.dropout(output)
        return self.output(output)


def count_parameters(model):
    """Number of trainable numbers in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """Builds the model for the real character vocabulary and runs one batch through it."""
    from src.baseline.splits import training_sentences
    from src.model.data import build_char_vocab, make_loader

    train = training_sentences("mixed")
    char_vocab = build_char_vocab(train)
    model = HierarchicalBiLSTM(len(char_vocab))
    print(model)
    print(f"{count_parameters(model):,} parameters")

    chars, labels, word_mask = next(iter(make_loader(train[:8], char_vocab, batch_size=8)))
    scores = model(chars, word_mask)
    loss = nn.CrossEntropyLoss(ignore_index=-100)(scores.reshape(-1, NUM_LABELS), labels.reshape(-1))
    print("chars", tuple(chars.shape), "-> scores", tuple(scores.shape), f"loss {loss.item():.3f}",
          f"(an untrained model should be near ln 4 = {torch.log(torch.tensor(4.0)).item():.3f})")


if __name__ == "__main__":
    main()
