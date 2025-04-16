from __future__ import absolute_import
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import average_precision_score


def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def cmc(distmat, query_ids=None, gallery_ids=None, topk=100,
        single_gallery_shot=False,
        first_match_break=False):
    distmat = distmat.cpu().numpy()
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)

    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)

    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(m):
        if list(query_ids) == list(gallery_ids):
            valid = (np.arange(n)[indices[i]] != np.arange(m)[i])
        else:
            valid = None
        if not np.any(matches[i, valid]): continue
        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk: break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    return ret.cumsum() / num_valid_queries


def mean_ap(distmat, query_ids=None, gallery_ids=None,):
   
    distmat = distmat.cpu().numpy()
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)

    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute AP for each query
    aps = []
    for i in range(m):
        # Filter out the same img
        if list(query_ids) == list(gallery_ids):
            valid = (np.arange(n)[indices[i]] != np.arange(m)[i])
        else:
            valid = None
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true): continue
        aps.append(average_precision_score(y_true, y_score))
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps)

def mean_ap_fct(
    distance_matrix: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Get pair-wise cosine distances.

    :param distance_matrix: pairwise distance matrix between embeddings of gallery and query sets, shape = (n, n)
    :param labels: labels for the query data (assuming the same as gallery), shape = (n,)

    :return: mean average precision (float)
    """
    distance_matrix = distance_matrix
    m, n = distance_matrix.shape
    assert m == n
    labels = np.asarray(labels)
    # Sort and find correct matches
    distance_matrix, gallery_matched_indices = torch.sort(distance_matrix, dim=1)
    distance_matrix = distance_matrix.cpu().numpy()
    gallery_matched_indices = gallery_matched_indices.cpu().numpy()

    truth_mask = labels[gallery_matched_indices] == labels[:, None]
    #truth_mask = truth_mask.cpu().numpy()

    # Compute average precision for each query
    average_precisions = list()
    for query_index in range(n):

        valid_sorted_match_indices = (
            gallery_matched_indices[query_index, :] != query_index
        )
        y_true = truth_mask[query_index, valid_sorted_match_indices]
        y_score = -distance_matrix[query_index][valid_sorted_match_indices]
        if not np.any(y_true):
            continue  # if a query does not have any match, we exclude it from mAP calculation.
        average_precisions.append(average_precision_score(y_true, y_score))
    return np.mean(average_precisions)
