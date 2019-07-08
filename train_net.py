import mxnet as mx
from mxnet import gluon,nd,lr_scheduler,autograd
from mxnet.gluon import nn
from mxnet.gluon import loss as gloss
import config as cfg


import copy

import math,time
import numpy as np
import matplotlib.pyplot as plt

class NoamOpt:
    def __init__(self,model_size,factor,warmup=5):
        self.model_size = model_size
        self.warmup = warmup
        self.factor = factor
        self.step = 0
        return
    def update(self, step = None):
        if step is None:
            self.step += 1
        else:
            self.step += step
        return self.factor * (self.model_size ** (-0.5) * min(self.step ** (-0.5), self.step * self.warmup ** (-1.5)))


if cfg.DEBUG_ON:
    opts = [NoamOpt(512, 1, 4000),
            NoamOpt(512, 1, 8000),
            NoamOpt(256, 1, 4000)]
    plt.plot(np.arange(1, 20000), [[opt.update(i) for opt in opts] for i in range(1, 20000)])
    plt.legend(["512:4000", "512:8000", "256:4000"])
   # plt.show()


class LabelSmoothing(nn.Block):
    "Implement label smoothing."

    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = gloss.KLDivLoss(from_logits=False)
        #self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        #self.true_dist = None

    def forward(self, x, target):
        assert x.shape[1] == self.size #sequence length

        with autograd.pause():
            true_dist = nd.zeros_like(x) + self.smoothing / (self.size - 2)
            target_mask = nd.zeros_like(true_dist)
            for r, c in enumerate(target):
                target_mask[r,c] = 1
            true_dist = nd.where(target_mask, nd.zeros_like(true_dist) + self.confidence, true_dist)
            # true_dist[:, self.padding_idx] = 0
            # mask = nd.equal(target,self.padding_idx)
            #
            # if len(mask.shape) > 0:
            #     true_dist = nd.where( nd.squeeze(mask), nd.zeros_like(true_dist) ,true_dist )

        #self.true_dist = true_dist
        return self.criterion(x, true_dist.as_in_context(cfg.ctx))

if cfg.DEBUG_ON:
    crit = LabelSmoothing(5, 0, 0.4)
    predict = nd.array([[0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0]])
    v = crit( nd.log1p(predict), nd.array([2,1,0]))

    # Show the target distributions expected by the system.
    plt.imshow(crit.true_dist.asnumpy())

if cfg.DEBUG_ON:
    crit = LabelSmoothing(5, 0, 0.1)
    def loss(x):
        d = x + 3 * 1.0
        predict = nd.array([[0, x / d, 1 / d, 1 / d, 1 / d],])
        return crit( nd.log1p(predict), nd.array([1]))
    plt.figure()
    L = [loss(x).asnumpy()[0] for x in range(1, 100)]

    plt.plot(np.arange(1, 100),L)


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, train_mode):
        x = self.generator(x)
        x = nd.reshape(x, (-1, x.shape[-1]))
        y = nd.reshape(y,(-1,))
        #loss = self.criterion(x,y).sum() / norm
        loss = self.criterion(x,y).mean()
        if train_mode:
            loss.backward()
        #return loss[0] * norm
        return loss[0]

import gluonnlp as nlp
import data_gen as datagen
def run_epoch(data_iter, model, loss_compute, generator = None, trainer = None, lr_sch=None):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = []
    tokens = 0
    if trainer is None:
        train_mode = False
    else:
        train_mode = True
    output = []
    for i, batch in enumerate(data_iter):
        if isinstance(data_iter, nlp.data.ShardedDataLoader) or isinstance(data_iter,gluon.data.DataLoader):
            batch = datagen.Batch( batch[0].astype(np.float32), batch[1].astype(np.float32), cfg.pad_val )
        if lr_sch and trainer:
            lr = lr_sch.update()
            trainer.set_learning_rate(lr)
        with autograd.record(train_mode):
            out = model(batch.src.as_in_context(cfg.ctx), batch.trg.as_in_context(cfg.ctx),batch.src_mask.as_in_context(cfg.ctx), batch.trg_mask.as_in_context(cfg.ctx))
            loss = loss_compute(out, batch.trg_y.as_in_context(cfg.ctx), train_mode)
        total_loss.append( loss.asnumpy()[0] )
        if generator:
            output.append( generator(out).as_in_context(mx.cpu(0)) )
        #total_tokens += batch.ntokens
        tokens += batch.ntokens
        if trainer:
            trainer.step(1)
        if i % 50 == 1:
            elapsed = time.time() - start
            LR = 0
            if trainer:
                LR = trainer.learning_rate
            print("Epoch Step: %d LR: %f Loss: %f Tokens per Sec: %f" %
                    (i,LR, loss.asnumpy()[0], tokens.asnumpy()[0] / elapsed))
            start = time.time()
            tokens = 0
    return sum(total_loss) / len(total_loss), output



if cfg.DEBUG_ON:
    plt.show()