import mxnet as mx
from mxnet import gluon,nd
from mxnet.gluon import nn
from mxnet.gluon import loss as gloss
import config as cfg

import math,pdb
import numpy as np
import matplotlib.pyplot as plt
import seaborn





class Embeddings(nn.Block):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        with self.name_scope():
            self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        return

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

if cfg.DEBUG_ON:
    net = Embeddings(cfg.D_MODEL, cfg.VOCAB_SIZE)
    net.initialize()
    x = nd.random.uniform(0,10,shape=(10,9))
    y = net(x)
    print("embedding debug")
    print("\tin-out:(batch-size, sequence-length, embedding-dim)",x.shape, y.shape)



def attention(query, key, value, mask=None, dropout=None):
    # Q * K.transpose() * value
    assert(len(query.shape) == 3)
    assert (len(key.shape) == 3)
    assert (len(value.shape) == 3)
    d_model = query.shape[-1]

    scores = nd.batch_dot(query, key, transpose_b=True) / math.sqrt(d_model)

    if mask is not None:
        val = nd.ones(scores.shape, ctx=cfg.ctx) * (-1e9)
        scores = nd.where(mask == 1, scores, val)
    p_attn = nd.softmax(scores, axis=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return nd.batch_dot(p_attn, value), p_attn

if cfg.DEBUG_ON:
    x = nd.uniform(0,1,shape=(1,9,512))
    out, attn = attention(x,x,x)
    print('attention debug:')
    print("\t input, out, attn: ",x.shape, out.shape, attn.shape)
   # print(attn[0,:,:])


class MultiHeadedAttention(nn.Block):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert (d_model % h == 0)
        with self.name_scope():
            self.d_k = d_model // h
            self.h = h

            self.linears_0 = nn.Dense(in_units=d_model, units=d_model,flatten=False)
            self.linears_1 = nn.Dense(in_units=d_model, units=d_model,flatten=False)
            self.linears_2 = nn.Dense(in_units=d_model, units=d_model,flatten=False)
            self.linears_3 = nn.Dense(in_units=d_model, units=d_model,flatten=False)
            #self.attn = None  # ????????????????
            self.dropout = nn.Dropout(dropout)
        return

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            if mask.shape[1] == 1: #encoding otherwise decoder
                #mask = nd.expand_dims(nd.squeeze(mask),-1) ##!!!!!!!!
                mask = nd.tile(mask, reps=(1,query.shape[1],1 ))
        bs = query.shape[0]

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #1) run linear transform from d_model to d_model
        #2) reshape and transpose to split input h heads
        query = nd.transpose(nd.reshape(self.linears_0(query), (bs, -1, self.h, self.d_k)), (0, 2, 1, 3))
        key  =  nd.transpose(nd.reshape(self.linears_1(key),   (bs, -1, self.h, self.d_k)), (0, 2, 1, 3))
        value = nd.transpose(nd.reshape(self.linears_2(value), (bs, -1, self.h, self.d_k)), (0, 2, 1, 3))


        #x = nd.zeros(value.shape)
        #for h in range(self.h):
        #    x[:,h,:,:],_ = attention(query[:,h,:,:], key[:,h,:,:], value[:,h,:,:], mask=mask, dropout=self.dropout)
        query,key,value = nd.reshape(query,(bs * self.h,-1,self.d_k)), nd.reshape(key,(bs * self.h,-1,self.d_k)), nd.reshape(value,(bs * self.h,-1,self.d_k))
        mask = nd.tile(mask, reps = (self.h,1,1))
        x, _ = attention(query, key, value, mask = mask, dropout=self.dropout)
        x = nd.reshape(x, (bs, self.h, -1, self.d_k))
        x = nd.reshape(nd.transpose(x, (0, 2, 1, 3)), (bs, -1, self.h * self.d_k))
        return self.linears_3(x)


if cfg.DEBUG_ON:
    x = nd.uniform(0,1,shape=(1,9,512))
    net = MultiHeadedAttention(h=8, d_model=512)
    net.initialize()
    y = net(x,x,x)
    print('multi-head attention debug:')
    print("\t input, out: ",x.shape, y.shape)


class PositionwiseFeedForward(nn.Block):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        with self.name_scope():
            self.w_1 = nn.Dense(in_units=d_model, units=d_ff, flatten=False)
            self.w_2 = nn.Dense(in_units=d_ff, units=d_model, flatten=False)
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(nd.relu(self.w_1(x))))



class LayerNorm(nn.Block):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        with self.name_scope():
            self.gamma = gluon.Parameter("gamma", shape=features,init=mx.initializer.Constant(1.0))
            self.beta = gluon.Parameter("beta", shape=features,init=mx.initializer.Constant(0.0))
        self.gamma.initialize(ctx=cfg.ctx)
        self.beta.initialize(ctx=cfg.ctx)
        self.eps = eps
        return

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        std = ((x - mean) ** 2).mean(axis=-1, keepdims=True).sqrt()
        return self.gamma.data() * (x - mean) / (std + self.eps) + self.beta.data()

if cfg.DEBUG_ON:
    net = LayerNorm(cfg.D_MODEL)
    net.initialize()
    x = nd.random.uniform(0,1,shape=(cfg.D_MODEL,cfg.D_MODEL))
    y = net(x)
    print("layer norm debug")
    print("\t in-out",x.shape, y.shape)

class SublayerConnection(nn.Block):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        with self.name_scope():
            self.norm = LayerNorm(size)
            self.dropout = nn.Dropout(dropout)
        return

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Block):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        with self.name_scope():
            self.self_attn = self_attn
            self.feed_forward = feed_forward
            self.sublayer_0 = SublayerConnection(size, dropout)
            self.sublayer_1 = SublayerConnection(size, dropout)
            self.size = size
        return

    def forward(self, x, mask):
        x = self.sublayer_0(x, lambda xx: self.self_attn(xx, xx, xx, mask))
        return self.sublayer_1(x, self.feed_forward)

if cfg.DEBUG_ON:
    ff = PositionwiseFeedForward(cfg.D_MODEL, cfg.D_FF)
    net = EncoderLayer(cfg.D_MODEL, MultiHeadedAttention(cfg.H, cfg.D_MODEL), ff, 0.5)
    #ff.initialize()
    net.initialize()
    x = nd.random.uniform(0,1,shape = (10,9,512))
    y = net(x, None)
    print("EncoderLayer debug:")
    print("\t input, output", x.shape, y.shape)


def make_encode_block():
    ff = PositionwiseFeedForward(cfg.D_MODEL, cfg.D_FF)
    net = EncoderLayer(cfg.D_MODEL, MultiHeadedAttention(cfg.H, cfg.D_MODEL), ff, 0.5)
    return net

class Encoder(nn.Block):
    def __init__(self):
        super(Encoder, self).__init__()
        with self.name_scope():
            self.layer_0 = make_encode_block()
            self.layer_1 = make_encode_block()
            if cfg.N == 6:
                self.layer_2 = make_encode_block()
                self.layer_3 = make_encode_block()
                self.layer_4 = make_encode_block()
                self.layer_5 = make_encode_block()
            self.norm = LayerNorm(cfg.D_MODEL)
        return

    def forward(self, x, mask):
        x = self.layer_0(x, mask)
        x = self.layer_1(x, mask)
        if cfg.N == 6:
            x = self.layer_2(x, mask)
            x = self.layer_3(x, mask)
            x = self.layer_4(x, mask)
            x = self.layer_5(x, mask)
        return self.norm(x)


if cfg.DEBUG_ON:
    net = Encoder()
    net.initialize()
    x = nd.random.uniform(0,1,shape = (10,9,512))
    y = net(x, None)
    print("Encoder debug:")
    print("\t input, output", x.shape, y.shape)


class PositionalEncoding(nn.Block):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute the positional encodings once in log space.
        pe = nd.zeros((max_len, d_model),ctx = cfg.ctx)
        position = nd.expand_dims(nd.arange(0, max_len), 1)
        div_term = nd.exp(nd.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = nd.sin(position * div_term)
        pe[:, 1::2] = nd.cos(position * div_term)
        pe = nd.expand_dims(pe,0)
        self.pe = pe #register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)

if cfg.DEBUG_ON:
    plt.figure(figsize=(15, 5))
    pe = PositionalEncoding(20, 0)
    y = pe.forward(nd.zeros((1, 100, 20)))
    plt.plot(np.arange(100), y[0, :, 4:8].asnumpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])



class DecoderLayer(nn.Block):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer_0 = SublayerConnection(size, dropout)
        self.sublayer_1 = SublayerConnection(size, dropout)
        self.sublayer_2 = SublayerConnection(size, dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer_0(x, lambda xx: self.self_attn(xx, xx, xx, tgt_mask))
        x = self.sublayer_1(x, lambda xx: self.src_attn(xx, m, m, src_mask))
        return self.sublayer_2(x, self.feed_forward)


def make_decode_block(dropout = 0.1):
    self_attn = MultiHeadedAttention(cfg.H, cfg.D_MODEL)
    src_attn = MultiHeadedAttention(cfg.H, cfg.D_MODEL)
    ff = PositionwiseFeedForward(cfg.D_MODEL,cfg.D_FF,dropout=dropout)
    return DecoderLayer(cfg.D_MODEL,self_attn,src_attn,ff,dropout=dropout)

class Decoder(nn.Block):
    def __init__(self):
        super(Decoder,self).__init__()
        with self.name_scope():
            self.layer_0 = make_decode_block()
            self.layer_1 = make_decode_block()
            if cfg.N == 6:
                self.layer_2 = make_decode_block()
                self.layer_3 = make_decode_block()
                self.layer_4 = make_decode_block()
                self.layer_5 = make_decode_block()
            self.norm = LayerNorm(cfg.D_MODEL)
        return
    def forward(self, x, memory, src_mask, target_mask):
        x = self.layer_0(x,memory, src_mask, target_mask)
        x = self.layer_1(x, memory, src_mask, target_mask)
        if cfg.N == 6:
            x = self.layer_2(x, memory, src_mask, target_mask)
            x = self.layer_3(x, memory, src_mask, target_mask)
            x = self.layer_4(x, memory, src_mask, target_mask)
            x = self.layer_5(x, memory, src_mask, target_mask)
        return self.norm(x)

if cfg.DEBUG_ON:
    net = Decoder()
    net.initialize()
    x = nd.random.uniform(0,1,shape = (10,9,512))
    y = net(x, x, None, None)
    print("Decoder debug:")
    print("\t input, output", x.shape, y.shape)


class EncoderDecoder(nn.Block):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Block):
    "Define standard linear + softmax generation step." #mapping to tgt vocab
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Dense(in_units=d_model, units=vocab,flatten=False)

    def forward(self, x):
        return nd.log_softmax(self.proj(x), axis=-1)

def make_model(dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    src_embeding = nn.Sequential()
    src_embeding.add(
        Embeddings(cfg.D_MODEL, cfg.SRC_VOCAB_SIZE),
        PositionalEncoding(cfg.D_MODEL, dropout) )
    tgt_embeding = nn.Sequential()
    tgt_embeding.add(
        Embeddings(cfg.D_MODEL, cfg.TGT_VOCAB_SIZE),
        PositionalEncoding(cfg.D_MODEL, dropout)
    )
    model = EncoderDecoder( Encoder(),Decoder(),src_embeding, tgt_embeding,  Generator(cfg.D_MODEL, cfg.TGT_VOCAB_SIZE))

    param_dict = model.collect_params()
    for k in param_dict:
        if len(param_dict[k].shape) > 1:
            param_dict[k].initialize(mx.init.Xavier(), force_reinit=True)
        else:
            param_dict[k].initialize(mx.init.Constant(0), force_reinit=True)
    return model

if cfg.DEBUG_ON:
    tmp_model = make_model()
    print(tmp_model)


if cfg.DEBUG_ON:
    plt.show()