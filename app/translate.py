import torch
import re

from mask import generate_square_subsequent_mask
from vocab import *

DEVICE = 'cpu'

# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate_sentence(model: torch.nn.Module, src_sentence: str, src_transform, vocab_transform_trg):
    print("Translating sentence")
    model.eval()
    src = src_transform(src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()

    pred_tokens = vocab_transform_trg.lookup_tokens(list(tgt_tokens.cpu().numpy()))

    pred_str = ""
    processed_tokens = []
    special_tokens_pattern = r'\W+'
    append_to_prev_token = False
    for i in range(len(pred_tokens)):
      # token that is ONLY special characters (e.g. ":", "-")
      if re.fullmatch(special_tokens_pattern, pred_tokens[i]):
        # append special token to previous processed token
        # this way, we can form long tokens like IX-loc:i
        token = processed_tokens[-1] + pred_tokens[i]
        processed_tokens[-1] = token
        append_to_prev_token = True
      # we just processed a special character and want to append current token to that token
      # e.g. IX- -> IX-loc
      elif append_to_prev_token:
        processed_tokens[-1] = processed_tokens[-1] + pred_tokens[i]
        append_to_prev_token = False
      else:
        processed_tokens.append(pred_tokens[i])
    pred_str = " ".join(processed_tokens)
    return pred_str.replace("<bos>", "").replace("<eos>", "")