
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import spacy
import json

import numpy as np
from numpy.linalg import norm

from torchtext.data.utils import get_tokenizer
from model import Seq2SeqTransformer

from sentence_transformers import SentenceTransformer

from transform import get_transformation_fn, tensor_transform
from translate import *

app = Flask(__name__)
CORS(app)

with open('/app/gloss_to_video_allLocal.json', 'r') as f:
  gloss_to_video = json.load(f)

with open('/app/asl_dict_text_gloss_with_custom_entries.json', 'r') as f:
  asl_dict_text_gloss = json.load(f)

nlp = spacy.load('en_core_web_sm')

token_transform = get_tokenizer('spacy')

model_st = SentenceTransformer('all-MiniLM-L6-v2')

# NCSLGR dataset text -> gloss
# vocab_transform_src_ncslgr = torch.load('ncslgr_augmented_rare_token_replacement_vocab_transform_src_text_to_gloss.pth')
# vocab_transform_trg_ncslgr = torch.load('ncslgr_augmented_rare_token_replacement_vocab_transform_trg_text_to_gloss.pth')
# transformer_ncslgr_text_gloss_state = torch.load('transformer_ncslgr_augmented_rare_token_replacement_text_to_gloss_90epochs.pt', map_location=torch.device('cpu'))

# ASLG PC12 dataset text -> gloss
# vocab_transform_src_aslg_pc12 = torch.load('aslg_pc12_vocab_transform_src_text_to_gloss.pth')
# vocab_transform_trg_aslg_pc12 = torch.load('aslg_pc12_vocab_transform_trg_text_to_gloss.pth')
# transformer_aslg_pc12_text_gloss_state = torch.load('transformer_aslg_text_to_gloss.pt', map_location=torch.device('cpu'))

# NCSLGR dataset w/ dictionary words text -> gloss
# vocab_transform_src_ncslgr_use_dict = torch.load('ncslgr_augmented_use_asl_dict_vocab_transform_src_text_to_gloss.pth')
# vocab_transform_trg_ncslgr_use_dict = torch.load('ncslgr_augmented_use_asl_dict_vocab_transform_trg_text_to_gloss.pth')
# transformer_ncslgr_use_dict_text_gloss_state = torch.load('transformer_ncslgr_augmented_use_asl_dict_text_to_gloss_90epochs.pt', map_location=torch.device('cpu'))

# NCSLGR dataset w/ dictionary words & adding word text -> gloss
vocab_transform_src_ncslgr_use_dict_add_word = torch.load('ncslgr_augmented_use_asl_dict_add_word_2_vocab_transform_src_text_to_gloss_pt3.pth')
vocab_transform_trg_ncslgr_use_dict_add_word = torch.load('ncslgr_augmented_use_asl_dict_add_word_2_vocab_transform_trg_text_to_gloss_pt3.pth')
transformer_ncslgr_use_dict_add_word_text_gloss_state = torch.load('transformer_ncslgr_augmented_use_asl_dict_add_word_text_to_gloss_2_pt3_40epochs.pt', map_location=torch.device('cpu'))

torch.manual_seed(0)

# NCSLGR
# SRC_VOCAB_SIZE = len(vocab_transform_src_ncslgr)
# TGT_VOCAB_SIZE = len(vocab_transform_trg_ncslgr)
# EMB_SIZE = 512
# NHEAD = 8
# FFN_HID_DIM = 512
# BATCH_SIZE = 128
# NUM_ENCODER_LAYERS = 3
# NUM_DECODER_LAYERS = 3

# transformer_text_gloss_ncslgr = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
#                                  NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

# transformer_text_gloss_ncslgr.load_state_dict(transformer_ncslgr_text_gloss_state)

# ASLG PC12
# SRC_VOCAB_SIZE = len(vocab_transform_src_aslg_pc12)
# TGT_VOCAB_SIZE = len(vocab_transform_trg_aslg_pc12)
# EMB_SIZE = 512
# NHEAD = 8
# FFN_HID_DIM = 512
# BATCH_SIZE = 128
# NUM_ENCODER_LAYERS = 3
# NUM_DECODER_LAYERS = 3

# transformer_text_gloss_aslg_pc12 = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
#                                  NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

# transformer_text_gloss_aslg_pc12.load_state_dict(transformer_aslg_pc12_text_gloss_state)

# NCSLGR w/ dict
# SRC_VOCAB_SIZE = len(vocab_transform_src_ncslgr_use_dict)
# TGT_VOCAB_SIZE = len(vocab_transform_trg_ncslgr_use_dict)
# EMB_SIZE = 512
# NHEAD = 8
# FFN_HID_DIM = 512
# BATCH_SIZE = 128
# NUM_ENCODER_LAYERS = 3
# NUM_DECODER_LAYERS = 3

# transformer_text_gloss_ncslgr_use_dict = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
#                                  NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

# transformer_text_gloss_ncslgr_use_dict.load_state_dict(transformer_ncslgr_use_dict_text_gloss_state)

# NCSLGR w/ dict & add word

SRC_VOCAB_SIZE = len(vocab_transform_src_ncslgr_use_dict_add_word)
TGT_VOCAB_SIZE = len(vocab_transform_trg_ncslgr_use_dict_add_word)
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer_text_gloss_ncslgr_use_dict_add_word = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

transformer_text_gloss_ncslgr_use_dict_add_word.load_state_dict(transformer_ncslgr_use_dict_add_word_text_gloss_state)


# src_transform_ncslgr = get_transformation_fn(token_transform, vocab_transform_src_ncslgr, tensor_transform)
# src_transform_aslg_pc12 = get_transformation_fn(token_transform, vocab_transform_src_aslg_pc12, tensor_transform)
# src_transform_ncslgr_use_dict = get_transformation_fn(token_transform, vocab_transform_src_ncslgr_use_dict, tensor_transform)
src_transform_ncslgr_use_dict_add_word = get_transformation_fn(token_transform, vocab_transform_src_ncslgr_use_dict_add_word, tensor_transform)

# @app.route("/translate/text/ncslgr", methods=['POST'])
# def translate_text_ncslgr():
#     print("Translating with NCSLGR model")
#     sentence = request.json["sentence"]
#     pred_text= translate_sentence(transformer_text_gloss_ncslgr, sentence, src_transform_ncslgr, vocab_transform_trg_ncslgr)
#     return jsonify({'pred' : pred_text})

# @app.route("/translate/text/aslg", methods=['POST'])
# def translate_text_aslg_pc12():
#     print("Translating with ASLG model")
#     sentence = request.json["sentence"]
#     pred_text= translate_sentence(transformer_text_gloss_aslg_pc12, sentence, src_transform_aslg_pc12, vocab_transform_trg_aslg_pc12)
#     return jsonify({'pred' : pred_text})

# @app.route("/translate/text/ncslgr_use_dict", methods=['POST'])
# def translate_text_ncslgr_use_dict():
#     print("Translating with NCSLGR model with dict words")
#     sentence = request.json["sentence"]
#     pred_text= translate_sentence(transformer_text_gloss_ncslgr_use_dict, sentence, src_transform_ncslgr_use_dict, vocab_transform_trg_ncslgr_use_dict)
#     return jsonify({'pred' : pred_text})

@app.route("/translate/text/ncslgr_use_dict_add_word", methods=['POST'])
def translate_text_ncslgr_use_dict_add_word():
    print("Translating with NCSLGR model with dict words and adding word")
    sentence = request.json["sentence"]

    tokens = nlp(sentence)
    tokens_no_punct = [x.text for x in tokens if x.pos_ != "PUNCT"]
    tokens_no_punct_lemmas = [x.lemma_ for x in tokens if x.pos_ != "PUNCT"]
    sentence_tokens = " ".join(tokens_no_punct)
    sentence_tokens_lemmas = " ".join(tokens_no_punct_lemmas)
    if sentence_tokens in asl_dict_text_gloss or sentence_tokens.lower() in asl_dict_text_gloss or sentence_tokens_lemmas in asl_dict_text_gloss:
        glosses = asl_dict_text_gloss.get(sentence_tokens) or asl_dict_text_gloss.get(sentence_tokens.lower()) or asl_dict_text_gloss.get(sentence_tokens_lemmas)
        max_cs = None
        best_gloss  = None
        for gloss in glosses:
            cs = cosine_similarity(model_st.encode(gloss), model_st.encode(sentence))
            if max_cs is None or cs > max_cs:
                max_cs = cs
                best_gloss = gloss
        pred_text = best_gloss
    else:
        pred_text= translate_sentence(transformer_text_gloss_ncslgr_use_dict_add_word, sentence, src_transform_ncslgr_use_dict_add_word, vocab_transform_trg_ncslgr_use_dict_add_word)
    pred_list = [x for x in pred_text.split(" ") if x]
    gloss_links = get_video_links(pred_text)
    return jsonify({'pred' : pred_text,
                    'pred_list': pred_list,
                    'links': gloss_links})

def cosine_similarity(a, b):
    return np.dot(a,b)/(norm(a)*norm(b))

def get_video_links(glosses):
    gloss_list = [x for x in glosses.split(" ") if x]
    ret_json = {}
    for gloss in gloss_list:
        if "IX" in gloss:
            ret_json[gloss] = gloss_to_video["IX"]
        if "LEARN+AGENT" in gloss:
            ret_json[gloss] = gloss_to_video["STUDENTneut"]
        if "TEACH+AGENT" in gloss:
            ret_json[gloss] = gloss_to_video["TEACHER"]
        else:
            if gloss in gloss_to_video:
                ret_json[gloss] = gloss_to_video[gloss]
    #ret = ""
    ret = []
    for gloss, link in ret_json.items():
        ret.append({"gloss": gloss, "link": link})
        # if ret == "":
        #     ret = "{}{}{}".format(gloss, "^", link)
        # else:
        #     ret += ",{}{}{}".format(gloss, "^", link)
    #return jsonify({'links' : ret})
    return ret

# Health check route
@app.route("/isalive")
def is_alive():
    print("/isalive request")
    status_code = Response(status=200)
    return status_code

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)