from typing import Iterable, List
from torchtext.vocab import build_vocab_from_iterator

# helper function to yield list of tokens
def yield_tokens(dataset: Iterable, tokenizer, data_sample_idx: str) -> List[str]:
  for data_sample in dataset:
    if data_sample_idx in data_sample:
      yield tokenizer(data_sample[data_sample_idx])
    else:
      yield []

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

def get_vocab_transform(dataset, tokenizer, src_idx, trg_idx, special_symbols=special_symbols):
  src_vocab = build_vocab_from_iterator(yield_tokens(dataset, tokenizer, src_idx),
                                                      min_freq=1,
                                                      specials=special_symbols,
                                                      special_first=True)
  src_vocab.set_default_index(UNK_IDX)

  trg_vocab = build_vocab_from_iterator(yield_tokens(dataset, tokenizer, trg_idx),
                                                      min_freq=1,
                                                      specials=special_symbols,
                                                      special_first=True)
  trg_vocab.set_default_index(UNK_IDX)

  vocab_transform = {
      src_idx: src_vocab,
      trg_idx: trg_vocab
  }
  
  return vocab_transform