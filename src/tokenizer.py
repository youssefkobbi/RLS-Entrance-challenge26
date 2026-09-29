"""Character-level tokenizer: 26 letters + <eos> (27 tokens).

Implements encode/decode between words and letter-index sequences, and
batches words of different lengths (padding + ignore_index for the loss).

Conventions (used by data.py, train.py, generate.py, evaluate.py):

* ids 0..25 are the letters a..z, id 26 is <eos>.  No start token and no
  padding token in the vocabulary (27 tokens in total).
* For a word w of L letters the model INPUT is the L letters (no <eos>) and
  the TARGET is the L letters followed by <eos> (L + 1 tokens).  The logit
  produced at the last visual token predicts letter 1, the logit produced at
  letter i predicts letter i + 1 (or <eos> after the last letter).
* Batches are padded with PAD_ID (any valid id works; we use 0) in the input,
  and with IGNORE_INDEX (-100) in the target so that cross-entropy skips them.
"""

import string

import torch

LETTERS = string.ascii_lowercase
EOS_TOKEN = "<eos>"
VOCAB_SIZE = len(LETTERS) + 1  # 27
EOS_ID = len(LETTERS)  # 26
PAD_ID = 0  # value used to pad the *inputs*; never seen by the loss (see IGNORE_INDEX)
IGNORE_INDEX = -100  # F.cross_entropy(ignore_index=...) skips these target positions
MAX_LETTERS = 45  # longest word produced by generate_data.py


class CharTokenizer:
    def __init__(self):
        self.itos = list(LETTERS) + [EOS_TOKEN]  # id -> token
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}  # token -> id
        self.vocab_size = VOCAB_SIZE
        self.eos_id = EOS_ID
        self.pad_id = PAD_ID

    def encode(self, word, add_eos=True):
        """'red' -> [17, 4, 3, 26] (with <eos>) or [17, 4, 3] (without)."""
        try:
            ids = [self.stoi[ch] for ch in word]
        except KeyError as err:
            raise ValueError(f"character {err} of {word!r} is not in a-z") from None
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        """Ids -> word.  Stops at the first <eos>; ignores pad / ignore_index values."""
        letters = []
        for i in ids:
            i = int(i)
            if i == self.eos_id:
                break
            if 0 <= i < len(LETTERS):
                letters.append(LETTERS[i])
        return "".join(letters)

    def batch(self, words):
        """Pad words of different lengths into two tensors.

        Returns
            input_ids: (B, L_max)      letters only, padded with PAD_ID
            targets:   (B, L_max + 1)  letters + <eos>, padded with IGNORE_INDEX
        """
        b = len(words)
        l_max = max(len(w) for w in words)
        input_ids = torch.full((b, l_max), self.pad_id, dtype=torch.long)
        targets = torch.full((b, l_max + 1), IGNORE_INDEX, dtype=torch.long)
        for row, word in enumerate(words):
            letters = self.encode(word, add_eos=False)
            input_ids[row, : len(letters)] = torch.tensor(letters, dtype=torch.long)
            targets[row, : len(letters) + 1] = torch.tensor(letters + [self.eos_id], dtype=torch.long)
        return input_ids, targets


if __name__ == "__main__":
    tok = CharTokenizer()
    ids = tok.encode("largeredcircle")
    print(ids, "->", tok.decode(ids))
