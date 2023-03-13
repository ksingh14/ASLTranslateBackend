import torch

from vocab import PAD_IDX

DEVICE = 'cpu'

def generate_square_subsequent_mask(sz):
    # Create square matrix of True/False values where bottom triangular half of matrix and diagonal is True, rest False
    # e.x. 3x3 -> 
    # [[ True, False, False],
    #  [ True,  True, False],
    #  [ True,  True,  True]]
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    # Replace all False with -inf, replace all True with 0.0
    # e.x. 3x3 -> 
    # [[0., -inf, -inf],
    #  [0., 0., -inf],
    #  [0., 0., 0.]]
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    # e.x. 3x3 -> 
    # [[0., -inf, -inf],
    #  [0., 0., -inf],
    #  [0., 0., 0.]]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)

    # e.x. 3x3 -> 
    # [[0, 0, 0],
    #  [0, 0, 0],
    #  [0, 0, 0]]
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask