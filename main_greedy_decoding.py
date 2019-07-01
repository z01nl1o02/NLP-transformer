import mxnet as mx
from data_gen import data_gen,subsequent_mask
from modules import make_model
from train_net import NoamOpt, LabelSmoothing, SimpleLossCompute,run_epoch
import config as cfg
from mxnet.gluon import Trainer


V = cfg.VOCAB_SIZE
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model()
model.collect_params().reset_ctx(cfg.ctx)
model_opt = NoamOpt(cfg.D_MODEL, 1, 400)
trainer = Trainer(model.collect_params(),"Adam",{"beta1":0.9,"beta2":0.98})

for epoch in range(10):
    run_epoch(data_gen(V, 30, 20), model,SimpleLossCompute(model.generator, criterion), trainer=trainer, lr_sch=model_opt)
    test_loss = run_epoch(data_gen(V, 30, 5), model,SimpleLossCompute(model.generator, criterion), trainer=None, lr_sch=model_opt)
    print("test loss: ", test_loss.asnumpy()[0])


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = mx.nd.zeros((1,1),dtype=src.dtype,ctx=cfg.ctx) + start_symbol
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,ys, subsequent_mask(ys.shape[1]).astype(src.dtype).as_in_context(cfg.ctx))
        prob = model.generator(out[:, -1])
        next_word = mx.nd.argmax(prob, axis = 1)
        next_word = next_word[0]
        ys = mx.nd.concat(ys,mx.nd.zeros((1,1),dtype=src.dtype,ctx=cfg.ctx) + next_word, dim=1)
    return ys

src = mx.nd.array([[1,2,3,4,5,6,7,8,9,10]],ctx=cfg.ctx)
src_mask = mx.nd.ones((1, 1, 10),ctx=cfg.ctx)
result = greedy_decode(model, src, src_mask, max_len=10, start_symbol=1)
print(result.asnumpy())

