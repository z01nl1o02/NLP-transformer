import mxnet as mx
from copy_scripts import dataprocessor
from copy_scripts.utils import logging_config
from copy_scripts.bleu import compute_bleu

DEBUG_ON = False

ctx = mx.gpu()

D_MODEL = 512
H = 8

N = 6 #block in encoding/decoding

D_FF = 2048

VOCAB_SIZE = 50
SRC_VOCAB_SIZE = VOCAB_SIZE
TGT_VOCAB_SIZE = VOCAB_SIZE

batch_size = 64

pad_val = 999 #unk

src_lang, tgt_lang = "en","de"

src_max_len, tgt_max_len = 64, 64
dataset = 'newstest2014'
save_dir = 'custom_transformer_{}_{}_{}'.format(src_lang, tgt_lang, D_MODEL)