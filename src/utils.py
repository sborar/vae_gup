from __future__ import division

import itertools
import math
import scipy.misc

import numpy as np
import random
import copy
import pickle
import pandas as pd
import csv
import os
import sys
import torch
import shutil
import pickle
import numpy as np
import bottleneck as bn
from sklearn.metrics.pairwise import cosine_similarity

def save_weights_pkl(fname, weights):
    with open(fname, 'wb') as f:
        pickle.dump(weights, f, pickle.HIGHEST_PROTOCOL)

def load_weights_pkl(fname):
    with open(fname, 'rb') as f:
        weights = pickle.load(f)
    return weights

def get_parameters(model, bias=False):
    for k, m in model.named_parameters():
        if bias:
            if k.endswith('.bias'):
                yield m
        else:
            if k.endswith('.weight'):
                yield m


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def get_idx_of_top_k(X_pred,k):
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)[:, :k]
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    return idx_topk

# def get_idx_of_top_k_from_multiple(prediction_list, k):
#     batch_users = prediction_list[0].shape[0]
#     idx_topk_joint = None
#     topk_joint = None
#     for pred in prediction_list:
#         idx_topk_part = bn.argpartition(-pred, k, axis=1)[:, :k]
#         topk_part = pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part]
#         idx_topk_joint = np.concatenate((idx_topk_joint, idx_topk_part), axis=1) if idx_topk_joint is not None else idx_topk_part
#         topk_joint = np.concatenate((topk_joint, topk_part), axis=1) if topk_joint is not None else topk_part
#     idx_part = np.argsort(-topk_joint, axis=1)
#     idx_topk = idx_topk_joint[np.arange(batch_users)[:, np.newaxis], idx_part]
#     return [f7(i)[:k] for i in idx_topk]

def get_idx_of_top_k_concat(prediction_list, k):
    batch_users = prediction_list[0].shape[0]
    idx_topk_joint = None
    topk_joint = None
    for pred in prediction_list:
        idx_topk_part = bn.argpartition(-pred, k, axis=1)[:, :k]
        topk_part = pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part]
        idx_part = np.argsort(-topk_part, axis=1)
        idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
        idx_topk_joint = np.concatenate((idx_topk_joint, idx_topk), axis=1) if idx_topk_joint is not None else idx_topk
    return idx_topk_joint

def select(concatenated_recs,k, divisions):
    separated_recs = np.split(concatenated_recs, divisions)
    number_of_recs_from_each = int(k/divisions)
    result = []
    seen = set([])
    counter = 1
    for rec in separated_recs:
        recs_to_add = k if counter == divisions else int(number_of_recs_from_each*counter)
        i = 0
        while len(result) < recs_to_add:
            if rec[i] not in seen:
                result.append(rec[i])
                seen.add(rec[i])
            i = i+1
        counter = counter + 1
    return result

def select_randomly(concatenated_recs,k, divisions):
    separated_recs = np.split(concatenated_recs, divisions)
    number_of_recs_from_each = int(k/divisions)
    result = []
    seen = set([])
    counter = 1
    for rec in separated_recs:
        recs_to_add = k if counter == divisions else int(number_of_recs_from_each*counter)
        i = 0
        while len(result) < recs_to_add:
            if rec[i] not in seen:
                result.append(rec[i])
                seen.add(rec[i])
            i = i+1
        counter = counter + 1
    return result

def select_by_diversity(concatenated_recs,k, item_cs_matrix):
    unique_recs = set(concatenated_recs)
    diversity_score = {}
    for item in unique_recs:
        other_items = list(unique_recs-set([item]))
        score = sum(1- item_cs_matrix.loc[item][other_items])
        diversity_score[item]=score
    rel_score = {}
    index = 0
    for item in concatenated_recs:
        if item in rel_score:
            rel_score[item].append(index)
        else:
            rel_score[item] = [index]
        index = index +1
        if (index)%10==0:
            index = 0
    rel_score = {k:np.mean(v) for k,v in rel_score.items()}
    diverse_list = sorted(diversity_score.items(), key=lambda item: item[1], reverse=True)[:k]
    return [i[0] for i in sorted(diverse_list, key = lambda ele: rel_score[ele[0]])]


def get_idx_of_top_k_combine(prediction_list, item_cs_matrix, k):
    batch_users = prediction_list[0].shape[0]
    idx_topk_joint = None
    for pred in prediction_list:
        idx_topk_part = bn.argpartition(-pred, k, axis=1)[:, :k]
        topk_part = pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part]
        idx_part = np.argsort(-topk_part, axis=1)
        idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
        idx_topk_joint = np.concatenate((idx_topk_joint, idx_topk), axis=1) if idx_topk_joint is not None else idx_topk
    return [select_by_diversity(i , k, item_cs_matrix) for i in idx_topk_joint]

# borrowed from https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb
def NDCG_binary_at_k_batch(recs, heldout_batch, k=100):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = len(recs)
    # NDCG_list = []

    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))
    # final_tp = tp
    # for i in range(len(prediction_list)-1):
    #     final_tp = np.concatenate((final_tp, tp), axis=None)

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis], recs] * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(int(n), 2*k)]).sum() for n in heldout_batch.sum(axis=1)])
    NDCG = DCG / IDCG
    return NDCG
    #     NDCG_list.append(NDCG)
    #
    # return np.mean(NDCG_list, axis=0)


def ILD_at_k(recs, item_cs_matrix, k=100):
    pair_lists = [itertools.combinations(user, 2) for user in recs]
    list_size = len(recs[0])
    ilds = [sum([1 - item_cs_matrix.loc[pair[0]][pair[1]] for pair in pair_list])/float(list_size*(list_size-1)) for pair_list in pair_lists]
    return ilds

def EILD_at_k(prediction_list, heldout_batch, item_cs_matrix,k=100):
    idx_topk = get_idx_of_top_k_combine(prediction_list, k, item_cs_matrix)
    tp = 1. / np.log2(np.arange(2, k + 2))

    batch_users = prediction_list[0].shape[0]
    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk] * tp)
    eilds = []
    for user, items in enumerate(idx_topk):
        eild = 0
        DCG_user = DCG[user]
        for i, item1 in enumerate(items):
            ieild = 0
            inorm = 0
            if DCG_user[i] == 0:
                continue
            eild_list = (1 - item_cs_matrix.loc[item1][items])*DCG_user
            eild_list.pop(item1)
            DCG_copy = np.delete(DCG_user,i)
            ieild += sum(eild_list)
            inorm += sum(DCG_copy)
            if inorm > 0:
                eild += DCG_user[i] * ieild /inorm

        eilds.append(eild)
    return eilds

def get_unique_items(recs, k=100):
    total_recommended_itemset = set()
    [total_recommended_itemset.update(set(i)) for i in recs]
    return total_recommended_itemset

def temporal_diversity_at_k(recs1, recs2, k=100):
    items_l2_l1 = [len(set(l2)-set(l1))/k for l1,l2 in zip(recs1, recs2)]
    return items_l2_l1


def temporal_diversity_rep_at_k(recs1, recs2, item_cs_matrix, k=100):
    list_size = len(recs1[0])
    pair_lists = [itertools.product(l1, l2) for l1,l2 in zip(recs1, recs2)]
    tilds = [sum([1 - item_cs_matrix.loc[pair[0]][pair[1]] for pair in pair_list])/float(list_size*(list_size-1)) for pair_list in pair_lists]
    return tilds

# def temporal_diversity_at_k(prediction_list, k=100):
#     result = []
#     for prediction_list1, prediction_list2 in itertools.combinations(prediction_list,2):
#         idx_topk = get_idx_of_top_k(prediction_list1, k)
#         idx_topk1 = get_idx_of_top_k(prediction_list2, k)
#         items_l2_l1 = [len(set(l2)-set(l1))/k for l1,l2 in zip(idx_topk, idx_topk1)]
#         result.append(items_l2_l1)
#     return result


def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1) # top k
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0)
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall

def get_pairwise_cosine_similarity_eff_for_model_input(path):
    with open(path, 'rb') as f:
        movie_embs = pickle.load(f)
    items = range(len(movie_embs))
    embs = np.array([np.array(movie_embs[i]) for i in items])
    cosine_matrix = pd.DataFrame(cosine_similarity(embs), index = items, columns=items)
    return cosine_matrix
