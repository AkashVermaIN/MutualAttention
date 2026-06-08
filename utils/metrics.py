# utils/metrics.py
import torch, numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import defaultdict
import math

def topk_acc(output, target, topk=(1,5)):
    # output: logits tensor (N, C), target: (N,)
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)  # (N, maxk)
        pred = pred.t()  # (maxk, N)
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k * 100.0) / target.size(0))
        return res  # list of tensors [top1, top5]

def param_count_and_size(model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # model size on disk (approx) - serialize state_dict temporarily (memory safe)
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_bytes = buffer.tell()
    return n_params, size_bytes

def save_confusion_matrix(cm, classes, fname):
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(range(len(classes)), classes, rotation=90)
    plt.yticks(range(len(classes)), classes)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def compute_precision_recall_f1(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return p, r, f1

def multiclass_roc_auc(y_true, y_prob, n_classes):
    # y_prob shape (N, C)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    # compute macro-avg AUC
    auc = roc_auc_score(y_bin, y_prob, average='macro', multi_class='ovr')
    return auc

def compute_cosine_similarity_between_models(state1, state2):
    # flatten all learnable params into vectors
    v1 = []
    v2 = []
    for k in state1:
        if state1[k].dtype.is_floating_point:
            v1.append(state1[k].detach().cpu().reshape(-1))
            v2.append(state2[k].detach().cpu().reshape(-1))
    v1 = torch.cat(v1).float()
    v2 = torch.cat(v2).float()
    cos = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
    return cos

def pca_or_tsne(embeddings, method='pca', n_components=2):
    X = embeddings.copy()
    if method=='pca':
        p = PCA(n_components=n_components)
        return p.fit_transform(X)
    else:
        ts = TSNE(n_components=n_components, init='pca', perplexity=30)
        return ts.fit_transform(X)
