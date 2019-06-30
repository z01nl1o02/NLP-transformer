import mxnet as mx
from mxnet import nd
import numpy as np
import config as cfg

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return nd.array(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = nd.expand_dims(src != pad,-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = nd.expand_dims(tgt != pad,-2)
        #tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        tgt_mask = nd.logical_and(tgt_mask, subsequent_mask(tgt.shape[-1]).astype(tgt_mask.dtype))
        return tgt_mask

def data_gen(V, batch, nbatches):

    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = nd.array(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = data
        tgt = data
        yield Batch(src, tgt, 0)

if cfg.DEBUG_ON:
    for batch in data_gen(11,1,2):
        print(batch.src, batch.trg, batch.trg_y)




