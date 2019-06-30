import mxnet as mx
from mxnet import gluon,nd
from mxnet.gluon import nn
from mxnet.gluon import loss as gloss


import math,pdb
import numpy as np
import matplotlib.pyplot as plt
import seaborn
#seaborn.set_context(context="talk")
#%matplotlib inline

debug_on = True


# main framework
class EncoderDecoder(nn.Block):
    def __init__(self, encoder, decoder, src_embed, target_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.generator = generator
        return

    def forward(self, src, target, src_mask, target_mask):
        return self.decode(self.encode(src, src_mask), src_mask, target, target_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, target, target_mask):
        return self.decoder(self.target_embed(target), memory, src_mask, target_mask)

# linear+softmax generation
class Generator(nn.Block):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        with self.name_scope():
            self.proj = nn.Dense(in_units=d_model, units=vocab)
        self.proj.initialize()
        return

    def forward(self, x):
        return mx.nd.log_softmax(self.proj(x), axis=-1)

if debug_on:
    x = nd.random.uniform(shape=(2, 100))
    net = Generator(100, 10)
    y = net(x)
    print('x:', x.shape, 'y:', y.shape)
    print(x, y)


def calc_std(x, axis=-1, keepdims=True, mean=None):
    if mean is None:
        mean = x.mean(axis=axis, keepdims=keepdims)
    return np.sqrt(((x - mean) ** 2))


class LayerNorm(nn.Block):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        with self.name_scope():
            self.gamma = gluon.Parameter("gamma", shape=features)
            self.beta = gluon.Parameter("beta", shape=features)
        self.gamma.initialize(mx.initializer.Constant(1.0))
        self.beta.initialize(mx.initializer.Constant(0.0))

        self.eps = eps
        return

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        std = ((x - mean) ** 2).mean(axis=-1, keepdims=True).sqrt()
        return self.gamma.data() * (x - mean) / (std + self.eps) + self.beta.data()


if debug_on:
    x = nd.random.uniform(shape=(2, 2))
    net = LayerNorm(2)
    y = net(x)
    print(y.shape)
    print(x, y)


# Encoder
def clones(module, N):
    layers = nn.Sequential()
    for n in range(N):
        layers.add(module)
    return layers


class Encoder(nn.Block):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        with self.name_scope():
            # self.layers = clones(layer, N)
            self.layers = nn.Sequential()
            for _ in range(N):
                self.layers.add(layer)

            self.norm = LayerNorm(layer.size)
        return

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SublayerConnection(nn.Block):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        with self.name_scope():
            self.norm = LayerNorm(size)
            self.dropout = nn.Dropout(dropout)
        return

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


if 0:
    x = nd.random.uniform(shape=(2, 2))
    net = SublayerConnection(2, 0.5)
    y = net(x)
    print('x:', x.shape, 'y:', y.shape)
    print(x, y)


class EncoderLayer(nn.Block):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        with self.name_scope():
            self.self_attn = self_attn
            self.feed_forward = feed_forward
            # self.sublayer = clones(SublayerConnection(size, dropout),2)
            self.sublayer = nn.Sequential()
            for _ in range(2):
                self.sublayer.add(SublayerConnection(size, dropout))
            self.size = size
        return

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Block):
    def __init__(self, layer, N):
        super(Decoder,self).__init__()
        with self.name_scope():
            #self.layers = clones(layer,N)
            self.layers = nn.Sequential()
            for _ in range(N):
                self.layers.add( layer )
            self.norm = LayerNorm(N)
        return
    def forward(self, x, memory, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x,memory, src_mask, target_mask)
        return self.norm(x)


class DecoderLayer(nn.Block):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer,self).__init__()
        with self.name_scope():
            self.size = size
            self.self_attn = self_attn
            self.src_attn = src_attn
            self.feed_forward = feed_forward
            #self.sublayer = clones(SublayerConnection(size, dropout),3)
            self.sublayer = nn.Sequential()
            for _ in range(3):
                self.sublayer.add( SublayerConnection(size, dropout))
        return
    def forward(self, x, memory, src_mask, target_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x,x,x,target_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x,m,m,src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    attn_shape = (1,size,size)
    subsequent_mask = np.triu(np.ones(attn_shape),k=1).astype('uint8')
    return nd.array(subsequent_mask) == 0

if debug_on:
    plt.figure(figsize=(5,5))
    plt.imshow(subsequent_mask(20).asnumpy()[0])


def attention(query, key, value, mask=None, dropout=None):
    # Q * K.transpose() * value
    print("query:",query.shape)
    print("key:",key.shape)
    print("value:",value.shape)
    d_k = query.shape[-1]
    dims_query = len(query.shape)
    dims_key = len(key.shape)
    if dims_query == 3 and dims_key == 3:
        scores = nd.batch_dot(query, key, transpose_b=True) / math.sqrt(d_k)
        # print('scoure out ', scores.shape)
    else:
        tmpQ = nd.reshape(query, (0, query.shape[1] * query.shape[2], query.shape[3]))
        tmpK = nd.reshape(key, (0, key.shape[1] * key.shape[2], key.shape[3]))
        # print('tmp:',tmpQ.shape, tmpK.shape)
        scores = nd.batch_dot(tmpQ, tmpK, transpose_b=True) / math.sqrt(d_k)
        scores = nd.reshape(scores, (key.shape[0], key.shape[1], key.shape[2], -1))
        scores = nd.transpose(scores, (0, 2, 1, 3))  # !!!!!!!!!!!
        # print('score out ',query.shape, key.shape, scores.shape)

    if mask is not None:
        val = nd.ones(scores.shape) * (-1e9)
        # print(mask.shape, val.shape, scores.shape)
        # print('mask:',mask.shape)
        scores = nd.where(mask == 1, scores, val)
        p_attn = nd.softmax(scores, axis=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    if dims_query == 3 and dims_key == 3:
        return nd.batch_dot(p_attn, value), p_attn

    # print(p_attn.shape, value.shape)
    tmpATT = nd.reshape(p_attn, (0, p_attn.shape[1] * p_attn.shape[2], p_attn.shape[3]))
    tmpVAL = nd.reshape(value, (0, value.shape[1] * value.shape[2], value.shape[3]))
    # print('tmp:',tmpATT.shape, tmpVAL.shape)
    scores = nd.batch_dot(tmpATT, tmpVAL)
    scores = nd.reshape(scores, (value.shape[0], value.shape[1], value.shape[2], -1))
    scores = nd.transpose(scores, (0, 2, 1, 3))  # !!!!!!!!!!!
    # print('score out ',p_attn.shape, value.shape, scores.shape)
    return scores, p_attn


if debug_on:
    batchsize = 10
    x = mx.nd.random.uniform(shape=(batchsize, 2, 30))
    mask = mx.nd.zeros((batchsize, x.shape[1], x.shape[1]), dtype=np.int32)
    y0, y1 = attention(x, x, x, mask=mask)
    print("x:", x.shape, 'y0:', y0.shape, 'y1:', y1.shape)


class MultiHeadedAttention(nn.Block):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert (d_model % h == 0)
        with self.name_scope():
            self.d_k = d_model // h
            self.h = h
            # self.linears = clones(nn.Dense(in_units=d_model, units=d_model),4)
            self.linears = nn.Sequential()
            for _ in range(4):
                self.linears.add(nn.Dense(in_units=d_model, units=d_model))
            self.attn = None  # ????????????????
            self.dropout = nn.Dropout(dropout)
        self.linears.collect_params().initialize(mx.init.Xavier())
        self.dropout.initialize(mx.init.Xavier())
        return

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = nd.expand_dims(mask, axis=1)
        nbatches = query.shape[0]
        # print("---$$$$$$$$$$$$$$$$$$$$$$--")
        #pdb.set_trace()
        results = []
        for l, v in zip(self.linears, (query, key, value)):
            t = nd.reshape(v,(-1,v.shape[-1]))
            xx = l(t)
            results.append( nd.reshape(xx, v.shape) )
       # query, key, value = [nd.transpose(nd.reshape(l(x), (nbatches, -1, self.h, self.d_k)), (0, 2, 1, 3)) for l, x in
        #                     zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = nd.reshape(nd.transpose(x, (0, 2, 1, 3)), (nbatches, -1, self.h * self.d_k))
        return self.linears[-1](x)


if debug_on:
    h = 20
    d_k = 5
    batchsize = 10
    d_model = h * d_k
    x = nd.random.uniform(shape=(10, h, d_k))
    mask = mx.nd.zeros((batchsize, x.shape[1], x.shape[1]), dtype=np.int32)
    net = MultiHeadedAttention(h, d_model)
    y = net(x, x, x, mask)
    print('x:', x.shape, 'y:', y.shape)


class Embeddings(nn.Block):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        with self.name_scope():
            self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        return

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Block):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        with self.name_scope():
            self.dropout = nn.Dropout(dropout)
            pe = nd.zeros(shape=(max_len, d_model))
            position = nd.expand_dims(nd.arange(0, max_len), 1)
            div_term = nd.exp(nd.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
            # pdb.set_trace()
            pe[:, 0::2] = nd.sin(position * div_term)
            pe[:, 1::2] = nd.cos(position * div_term)
            pe = nd.expand_dims(pe, 0)
            # self.register_buffer('pe',pe)
            self.pe = pe
        return

    def forward(self, x):
        # x = x + Variable(self.pe[:,:x.size(1)], requires_grad = False)
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)


if debug_on:
    plt.figure(figsize=(15, 5))
    pe = PositionalEncoding(20, 0)
    x = nd.zeros(shape=(1, 100, 20))
    y = pe.forward(x)
    plt.plot(np.arange(100), y[0, :, 4:8].asnumpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])


class PositionwiseFeedForward(nn.Block):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        with self.name_scope():
            self.w_1 = nn.Dense(in_units=d_model, units=d_ff)
            self.w_2 = nn.Dense(in_units=d_ff, units=d_model)
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# import copy
import pdb


def make_model(src_vocab, target_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    #    c = copy.deepcopy
    self_attn_encoding, self_attn_decoding = MultiHeadedAttention(h, d_model), MultiHeadedAttention(h, d_model)
    src_attn_decoding = MultiHeadedAttention(h, d_model)

    ff_encoding, ff_decoding = PositionwiseFeedForward(d_model, d_ff, dropout), PositionwiseFeedForward(d_model, d_ff,
                                                                                                        dropout)
    position_src_embed, position_target_embed = PositionalEncoding(d_model, dropout), PositionalEncoding(d_model,
                                                                                                         dropout)

    src_embedings, target_embedings = nn.Sequential(), nn.Sequential()
    src_embedings.add(
        Embeddings(d_model, src_vocab), position_src_embed
    )
    target_embedings.add(
        Embeddings(d_model, target_vocab), position_target_embed
    )

    encoder = Encoder(EncoderLayer(d_model, self_attn_encoding, ff_encoding, dropout), N)
    decoder = Decoder(DecoderLayer(d_model, self_attn_decoding, src_attn_decoding, ff_decoding, dropout), N)
    generator = Generator(d_model, target_vocab)

    model = EncoderDecoder(
        encoder, decoder, src_embedings, target_embedings, generator)

    param_dict = model.collect_params()
    for k in param_dict:
        # print(k, param_dict[k])
        # param_dict[k].initialize(mx.init.Xavier(), force_reinit=True)
        if len(param_dict[k].shape) > 1:
            param_dict[k].initialize(mx.init.Xavier(), force_reinit=True)
        else:
            param_dict[k].initialize(mx.init.Constant(0), force_reinit=True)
    return model


if debug_on:
    tmp_model = make_model(10, 10, 2)


class Batch:
    def __init__(self, src, trg = None, pad = 0):
        self.src = src
        self.src_mask = nd.expand_dims(src != pad,-2)
        if trg is not None:
            self.trg = trg[:,:-1]
            self.trg_y = trg[:,1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).sum()
        return
    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = nd.expand_dims(tgt != pad, -2)
        #tgt_mask = tgt_mask & (subsequent_mask(tgt.shape[-1]).astype(tgt_mask.dtype))
        tgt_mask = nd.logical_and(tgt_mask, subsequent_mask(tgt.shape[-1]).astype(tgt_mask.dtype))
        return tgt_mask


from mxnet.gluon import Trainer


class NoamOpt:
    def __init__(self, model, model_size, factor, warmup):
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        if model is not None:
            self._trainer = Trainer(model.collect_params(), optimizer="Adam",
                                    optimizer_params={"wd": 1e-5, "beta1": 0.9, 'beta2': 0.98, 'epsilon': 1e-9})
        else:
            self._trainer = None
        return

    def step(self):
        self._step += 1
        rate = self.rate()
        self._rate = rate
        if self._trainer is not None:
            self._trainer.set_learning_rate(rate)
            self._trainer.step()
        return

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


opts = [NoamOpt(None, 512, 1, 4000),
        NoamOpt(None, 512, 1, 8000),
        NoamOpt(None, 256, 1, 4000)]
plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
plt.legend(["512:4000", "512:8000", "256:4000"])


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Block):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = gloss.KLDivLoss()  ##???
        self.padding_idx = padding_idx
        self.confidence = 1 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        return

    def forward(self, x, target):
        assert (x.size(1) == self.size)
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0, 0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))



import time
def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        with autograd.record(True):
            out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = nd.array(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = data
        tgt = data
        yield Batch(src, tgt, 0)


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        if self.opt is not None:
            loss.backward()
            self.opt.step()
        return loss.data[0] * norm


from mxnet import autograd
# Train the simple copy task.
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
adam = mx.optimizer.Adam(beta1=0.9, beta2=0.98, epsilon=1e-9)
adam.base_lr = 0
model_opt = NoamOpt(model,model.src_embed[0].d_model, 1, 400)


for epoch in range(10):
    run_epoch(data_gen(V, 30, 20), model, SimpleLossCompute(model.generator, criterion, model_opt) )
    print(run_epoch(data_gen(V, 30, 5), model, SimpleLossCompute(model.generator, criterion, None)))