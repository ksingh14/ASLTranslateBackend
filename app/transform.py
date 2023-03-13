import torch
from typing import List

from vocab import BOS_IDX, EOS_IDX

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))
    
def get_transformation_fn(*transforms):
  def apply_transformations(sentence: str):
    for transform in transforms:
      sentence = transform(sentence)
    return sentence
  return apply_transformations