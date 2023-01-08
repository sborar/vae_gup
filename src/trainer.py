# -*- coding: utf-8 -*-

import datetime
import math
import os
import time

import numpy as np
import scipy.io
import torch
from torch.autograd import Variable
import torch.nn.functional as f

import utils
from utils import AverageMeter
import tqdm


class Trainer(object):

    def __init__(self, cmd, cuda, model, optim=None,
                 train_loader=None, valid_loader=None, test_loader=None, log_file=None,
                 interval_validate=1, lr_scheduler=None,
                 start_step=0, total_steps=1e5, beta=0.05, start_epoch=0,
                 total_anneal_steps=200000, anneal_cap=0.2, do_normalize=True,
                 checkpoint_dir=None, checkpoint_path=None, result_dir=None, print_freq=1, result_save_freq=1,
                 checkpoint_freq=1,em_path= None):

        self.cmd = cmd
        self.cuda = cuda
        self.model = model

        self.optim = optim
        self.lr_scheduler = lr_scheduler

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.timestamp_start = datetime.datetime.now()

        if self.cmd == 'train':
            self.interval_validate = interval_validate

        self.start_step = start_step
        self.step = start_step
        self.total_steps = total_steps
        self.epoch = start_epoch

        self.do_normalize = do_normalize
        self.print_freq = print_freq
        self.checkpoint_freq = checkpoint_freq

        self.checkpoint_dir = checkpoint_dir

        self.checkpoint_path = checkpoint_dir + checkpoint_path + '.pth'

        self.total_anneal_steps = total_anneal_steps
        self.anneal_cap = anneal_cap

        self.cyc_anneal_array = self.frange_cycle_linear(7110)

        self.n20_all = []
        self.n10_max_va, self.n50_max_va, self.n100_max_va, self.r20_max_va, self.r50_max_va, self.agg10_max_va, self.agg50_max_va, self.td10_max_va, self.td50_max_va = 0, 0, 0, 0, 0, 0, 0, 0, 0
        self.n10_max_te, self.n50_max_te, self.n100_max_te, self.r20_max_te, self.r50_max_te, self.agg10_max_te, self.agg50_max_te, self.td10_max_te, self.td50_max_te = 0, 0, 0, 0, 0, 0, 0, 0, 0
        # self.sbert_em_path = "../item2vec/vae_cf-master/bol_data/cat_emb_multilingual2.pkl"
        # self.multi_em_path = "../item2vec/vae_cf-master/bol_data/cat_emb_multilingual.pkl"
        self.em_path = em_path
        # self.cs_matrix_bert = utils.get_pairwise_cosine_similarity_eff_for_model_input(self.sbert_em_path)
        self.cs_matrix_multi = utils.get_pairwise_cosine_similarity_eff_for_model_input(self.em_path)
        # self.wandb = wandb
        # ck_file_path
        # if checkpoint_path:
        #     self.model.load_state_dict(torch.load(self.checkpoint_path))

    def validate(self, cmd="valid"):
        assert cmd in ['valid', 'test']
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        self.model.eval()

        end = time.time()

        n10_list, n20_list, n5_list, r20_list, r50_list, eild10_sbert_list, eild10_multi_list, eild50_sbert_list, eild50_multi_list, ild10_sbert_list, ild5_multi_list, ild10_multi_list, ild20_multi_list, td5_list, td10_list, td20_list, tdrep5_list, tdrep10_list, tdrep20_list = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        agg20_set, agg10_set, agg5_set = set(), set(), set()

        loader_ = self.valid_loader if cmd == 'valid' else self.test_loader

        step_counter = 0
        for batch_idx, (data_tr, data_te, prof) in tqdm.tqdm(enumerate(loader_), total=len(loader_),
                                                             desc='{} check epoch={}'.format(
                                                                 'Valid' if cmd == 'valid' else 'Test',
                                                                 self.epoch), ncols=80, leave=False):
            step_counter = step_counter + 1

            if self.cuda:
                data_tr = data_tr.cuda()
                prof = prof.cuda()
            data_tr = Variable(data_tr)
            prof = Variable(prof)
            data_time.update(time.time() - end)
            end = time.time()

            with torch.no_grad():
                if self.model.__class__.__name__ == 'MultiVAE':
                    logits, KL, mu_q, std_q, epsilon, sampled_z = self.model.forward(data_tr, prof)
                    logits1, KL1, mu_q1, std_q1, epsilon1, sampled_z1 = self.model.forward(data_tr, prof)
                    logits2, KL2, mu_q2, std_q2, epsilon2, sampled_z2 = self.model.forward(data_tr, prof)
                    logits3, KL3, mu_q3, std_q3, epsilon3, sampled_z3 = self.model.forward(data_tr, prof)
                else:
                    logits = self.model.forward(data_tr)
                    logits1 = self.model.forward(data_tr)
                    logits2 = self.model.forward(data_tr)
                    logits3 = self.model.forward(data_tr)
                pred_val = logits.cpu().detach().numpy()
                pred_val[data_tr.cpu().detach().numpy().nonzero()] = -np.inf
                pred_val1 = logits1.cpu().detach().numpy()
                pred_val1[data_tr.cpu().detach().numpy().nonzero()] = -np.inf
                pred_val2 = logits2.cpu().detach().numpy()
                pred_val2[data_tr.cpu().detach().numpy().nonzero()] = -np.inf
                pred_val3 = logits3.cpu().detach().numpy()
                pred_val3[data_tr.cpu().detach().numpy().nonzero()] = -np.inf

                prediction_list = [pred_val, pred_val1]
                prediction_list1 = [pred_val2, pred_val3]

                recs10_1 = utils.get_idx_of_top_k_combine(prediction_list, self.cs_matrix_multi, k=10)
                recs10_2 = utils.get_idx_of_top_k_combine(prediction_list1, self.cs_matrix_multi, k=10)
                n10_list.append(utils.NDCG_binary_at_k_batch(recs10_1, data_te.numpy(), k=10))
                agg10_set = agg10_set.union(utils.get_unique_items(recs10_1, k=10))
                td10_list.append(utils.temporal_diversity_at_k(recs10_1, recs10_2, k=10))
                tdrep10_list.append(utils.temporal_diversity_rep_at_k(recs10_1, recs10_2, self.cs_matrix_multi, k=10))
                ild10_multi_list.append(utils.ILD_at_k(recs10_1, self.cs_matrix_multi, k=10))
        n10_list = np.concatenate(n10_list, axis=0)
        agg_10_mean = len(agg10_set) / pred_val.shape[1]
        td10_list = np.concatenate(td10_list, axis=0)
        tdrep10_list = np.concatenate(tdrep10_list, axis=0)
        ild10_multi_list = np.concatenate(ild10_multi_list, axis=0)

        print("epoch:", self.epoch)
        print("NDCG@10_mean: {} NDCG@10_std {}".format(np.mean(n10_list), np.std(n10_list) / np.sqrt(len(n10_list))))

        print("AGG@10_mean", agg_10_mean)
        print("TD@10_mean: {} TD@10_std: {}".format(np.mean(td10_list), np.std(td10_list) / np.sqrt(len(td10_list))))
        print("TILD@10_mean: {} TILD@10_std: {}".format(np.mean(tdrep10_list),
                                                        np.std(tdrep10_list) / np.sqrt(len(tdrep10_list))))
        print("ILD@10_mpnet_mean: {} ILD@10_mpnet_std: {}".format(np.mean(ild10_multi_list),
                                                                  np.std(ild10_multi_list) / np.sqrt(
                                                                      len(ild10_multi_list))))
        self.model.train()

    def frange_cycle_linear(self, n_iter, start=0.0, stop=0.2, n_cycle=4, ratio=0.5):
        L = np.ones(n_iter) * stop
        period = n_iter / n_cycle
        step = (stop - start) / (period * ratio)  # linear schedule

        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i + c * period) < n_iter):
                L[int(i + c * period)] = v
                v += step
                i += 1
        return L

    def train_epoch(self):
        cmd = "train"
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        self.model.train()

        end = time.time()
        for batch_idx, (data_tr, data_te, prof) in tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                                                             desc='Train check epoch={}'.format(self.epoch), ncols=80,
                                                             leave=False):
            self.step += 1

            if self.cuda:
                data_tr = data_tr.cuda()
                prof = prof.cuda()
            data_tr = Variable(data_tr)
            prof = Variable(prof)
            data_time.update(time.time() - end)
            end = time.time()

            if self.model.__class__.__name__ == 'MultiVAE':
                logits, KL, mu_q, std_q, epsilon, sampled_z = self.model.forward(data_tr, prof)
            else:
                logits = self.model.forward(data_tr)

            log_softmax_var = f.log_softmax(logits, dim=1)
            neg_ll = - torch.mean(torch.sum(log_softmax_var * data_tr, dim=1))  # do it only in places where its one
            l2_reg = self.model.get_l2_reg()

            if self.model.__class__.__name__ == 'MultiVAE':
                if self.total_anneal_steps > 0:
                    self.anneal = min(self.anneal_cap, 2. * self.step / self.total_anneal_steps)
                else:
                    self.anneal = self.anneal_cap
                loss = neg_ll + self.anneal * KL + l2_reg
                print("MultiVAE", self.epoch, batch_idx, loss.item(), neg_ll.cpu().detach().numpy(),
                      KL.cpu().detach().numpy(), l2_reg.cpu().detach().numpy() / 2, self.anneal, self.step,
                      self.optim.param_groups[0]['lr'])
            else:
                loss = neg_ll + l2_reg
                print("MultiDAE", self.epoch, batch_idx, loss.item(), neg_ll.cpu().detach().numpy(),
                      l2_reg.cpu().detach().numpy() / 2, self.step)

            # backprop
            self.model.zero_grad()
            loss.backward()
            self.optim.step()

            # if self.interval_validate > 0 and (self.step + 1) % self.interval_validate == 0:
            #     print("CALLING VALID", cmd, self.step, )
            #     self.validate()

    def train(self):
        max_epoch = 100
        for epoch in tqdm.trange(0, max_epoch, desc='Train', ncols=80):
            self.epoch = epoch
            self.lr_scheduler.step()
            self.train_epoch()
            if (self.epoch + 1) % 100 == 0:
                self.validate(cmd='valid')
                if epoch % self.checkpoint_freq:
                    torch.save(self.model.state_dict(), self.checkpoint_path)
