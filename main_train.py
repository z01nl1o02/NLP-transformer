# coding: utf-8
import data_gen as datagen
import config as cfg
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import Trainer
import logging
import random,argparse,os
from train_net import NoamOpt, LabelSmoothing, SimpleLossCompute,run_epoch
from modules import make_model

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)

parser = argparse.ArgumentParser(description='Neural Machine Translation Example.'
                                             'We train the Google NMT model')
parser.add_argument('--epochs', type=int, default=40, help='upper epoch limit')
parser.add_argument('--num_hidden', type=int, default=128, help='Dimension of the embedding '
                                                                'vectors and states.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers in the encoder'
                                                              ' and decoder')
parser.add_argument('--num_bi_layers', type=int, default=1,
                    help='number of bidirectional layers in the encoder and decoder')
parser.add_argument('--beam_size', type=int, default=4, help='Beam size')
parser.add_argument('--lp_alpha', type=float, default=1.0,
                    help='Alpha used in calculating the length penalty')
parser.add_argument('--lp_k', type=int, default=5, help='K used in calculating the length penalty')
parser.add_argument('--num_buckets', type=int, default=5, help='Bucket number')
parser.add_argument('--bucket_scheme', type=str, default='constant',
                    help='Strategy for generating bucket keys. It supports: '
                         '"constant": all the buckets have the same width; '
                         '"linear": the width of bucket increases linearly; '
                         '"exp": the width of bucket increases exponentially')
parser.add_argument('--bucket_ratio', type=float, default=0.0, help='Ratio for increasing the '
                                                                    'throughput of the bucketing')
parser.add_argument('--optimizer', type=str, default='adam', help='optimization algorithm')
parser.add_argument('--lr', type=float, default=1E-3, help='Initial learning rate')
parser.add_argument('--lr_update_factor', type=float, default=0.5,
                    help='Learning rate decay factor')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save_dir', type=str, default='out_dir',
                    help='directory path to save the final model and training log')


def score2sentence(scores, vocab):
    sentences = []
    for batch in scores:
        for sentence_inds in batch:
            sentence = []
            token_inds = np.argmax(sentence_inds, axis=-1)
            token_inds = [int(x.asnumpy()[0]) for x in token_inds]
            for token_ind in token_inds:
                token = vocab.idx_to_token[token_ind]
                sentence.append(token)
            sentences.append(sentence)
    return sentences

if __name__=="__main__":
    args = parser.parse_args()
    args.dataset = "TOY" #'IWSLT2015'
    #args.dataset = 'IWSLT2015'
    args.src_lang = cfg.src_lang
    args.tgt_lang = cfg.tgt_lang
    args.src_max_len = cfg.src_max_len
    args.tgt_max_len = cfg.tgt_max_len
    args.batch_size = cfg.batch_size
    args.test_batch_size = cfg.batch_size
    print(args)
    cfg.logging_config(args.save_dir)

    data_train, data_val, data_test, val_tgt_sentences, test_tgt_sentences, src_vocab, tgt_vocab \
        = cfg.dataprocessor.load_translation_data(dataset=args.dataset, bleu='tweaked', args=args)

    cfg.dataprocessor.write_sentences(val_tgt_sentences, os.path.join(args.save_dir, 'val_gt.txt'))
    cfg.dataprocessor.write_sentences(test_tgt_sentences, os.path.join(args.save_dir, 'test_gt.txt'))

    data_train = data_train.transform(lambda src, tgt: (src, tgt, len(src), len(tgt)), lazy=False)
    data_val = gluon.data.SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                                         for i, ele in enumerate(data_val)])
    data_test = gluon.data.SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                                          for i, ele in enumerate(data_test)])

    train_data_loader, val_data_loader, test_data_loader \
        = cfg.dataprocessor.make_dataloader(data_train, data_val, data_test, args,num_workers = -1)


    # for src, tgt, src_len, tgt_len in train_data_loader:
    #     for src_sentence, src_sentence_len in zip(src, src_len):
    #         if (src_sentence != cfg.pad_val).sum().asnumpy()[0] != int(src_sentence_len.asnumpy()[0]):
    #             print(src_sentence, src_sentence_len)
    # exit(0)
    cfg.SRC_VOCAB_SIZE, cfg.TGT_VOCAB_SIZE = len(src_vocab), len(tgt_vocab)
    cfg.VOCAB_SIZE = np.maximum(cfg.SRC_VOCAB_SIZE, cfg.TGT_VOCAB_SIZE)
    #V = cfg.VOCAB_SIZE
    criterion = LabelSmoothing(size=cfg.TGT_VOCAB_SIZE, smoothing=0.0)
    model = make_model()
    model.collect_params().reset_ctx(cfg.ctx)
    model_opt = NoamOpt(cfg.D_MODEL, 1, 400)
    trainer = Trainer(model.collect_params(), "Adam", {"beta1": 0.9, "beta2": 0.98})
    loss_func = SimpleLossCompute(model.generator, criterion)
    for epoch in range(args.epochs):
        run_epoch(train_data_loader, model, loss_func, trainer=trainer, lr_sch=model_opt)
        val_loss,val_eval = run_epoch(val_data_loader, model, loss_func, generator = model.generator,trainer=None, lr_sch=model_opt)
        val_tgt_sentences_eval = score2sentence(val_eval, tgt_vocab)
        valid_bleu_score, _, _, _, _ = cfg.compute_bleu([val_tgt_sentences], val_tgt_sentences_eval)
        print("val loss: ", val_loss, "val bleu: ",valid_bleu_score)