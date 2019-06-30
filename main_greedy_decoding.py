import mxnet as mx
from data_gen import data_gen
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
