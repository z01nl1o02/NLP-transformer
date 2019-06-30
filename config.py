import mxnet as mx
DEBUG_ON = False

ctx = mx.gpu()

D_MODEL = 512
H = 8

N = 6 #block in encoding/decoding

D_FF = 2048

VOCAB_SIZE = 11
SRC_VOCAB_SIZE = VOCAB_SIZE
TGT_VOCAB_SIZE = VOCAB_SIZE