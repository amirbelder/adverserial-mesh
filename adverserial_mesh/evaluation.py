import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import average_precision_score
# import ipdb

def plot_pr_cure(mpres, mrecs):
    pr_curve = np.zeros(mpres.shape[0], 10)
    for r in range(mpres.shape[0]):
        this_mprec = mpres[r]
        for c in range(10):
            pr_curve[r, c] = np.max(this_mprec[mrecs[r]>(c-1)*0.1])
    return pr_curve


def Eu_dis_mat_fast(X):
  aa = np.sum(np.multiply(X, X), 1)
  ab = X * X.T
  D = aa + aa.T - 2 * ab
  D[D < 0] = 0
  D = np.sqrt(D)
  D = np.maximum(D, D.T)
  return D


def calc_precision_recall(r):
  '''

  :param r:  ranked array of retrieved objects - '1' if we retrieved the correct label, '0' otherwise
  :return:
  '''
  num_gt = np.sum(r)   # total number of GT in class
  trec_precision = np.array([np.mean(r[:i+1]) for i in range(r.shape[0]) if r[i]])
  recall = [np.sum(r[:i+1]) / num_gt for i in range(r.shape[0])]
  precision = [np.mean(r[:i + 1]) for i in range(r.shape[0])]

  # interpolate it
  mrec = np.array([0.] + recall + [1.])
  mpre = np.array([0.] + precision + [0.])

  for i in range(len(mpre) - 2, -1, -1):
    mpre[i] = max(mpre[i], mpre[i + 1])

  i = np.where(mrec[1:] != mrec[:-1])[0] + 1  # Is this a must? why not sum all?
  ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i])   # area under the PR graph, according to information retrieval
  return trec_precision, mrec, mpre, ap


def calculate_map_auc(fts, lbls, models_ids=None, dis_mat=None):
  if dis_mat is None:
    dis_mat = Eu_dis_mat_fast(np.mat(fts))
  num = len(lbls)
  mAP = 0
  trec_precisions = []
  mrecs = []
  mpres = []
  aps = []
  visited = []
  for i in range(num):
    scores = dis_mat[:, i]
    # # TODO: test those lines, drop in performance is way too high
    if models_ids is not None:
      # For each unique model_id, calculate scores as mean distance over the number of models with same model_ids
      if models_ids[i] in visited:
        continue
      visited.append(models_ids[i])
      all_model_i = np.where(np.asarray(models_ids) == models_ids[i])[0]
      scores = np.mean(dis_mat[:, all_model_i], axis=-1)

    targets = (lbls == lbls[i]).astype(np.uint8)   # current lbl predictions
    sortind = np.argsort(scores, 0)   #[:top_k]
    if models_ids is not None:
      # ==== Exclude self-retrieved models (scale to scale and self) === #
      positions_to_keep = []
      cur_model_id = models_ids[i]
      for position, idx in enumerate(sortind):
        if models_ids[idx.item()] != cur_model_id:
          positions_to_keep.append(position)
      sortind = sortind[positions_to_keep,:]
      # =============================================================== #

      # ===== Choose minimal distance over multiscale shapes walks ==== #
      seen_model_ids = []
      positions_to_keep = []
      for position, idx in enumerate(sortind):
        cur_model_id = models_ids[idx.item()]
        if not cur_model_id in seen_model_ids:
          seen_model_ids.append(cur_model_id)
          positions_to_keep.append(position)
      sortind = sortind[positions_to_keep, :]
      # ============================================================== #
    # assert len(sortind) == 2467
    truth = targets[sortind]   # retrieved list, 1 if same class 0 otherwise
    res = calc_precision_recall(truth)
    trec_precisions.append(res[0])
    mrecs.append(res[1])
    mpres.append(res[2])
    aps.append(res[3])

  trec_precisions = np.concatenate(trec_precisions)
  mrecs = np.stack(mrecs)
  mpres = np.stack(mpres)
  # weighted sum of PR AUC per class
  aps = np.stack(aps)
  AUC = np.mean(aps)   # # according to 3D-ShapeNet   (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7298801)
  mAP = np.mean(trec_precisions)  # according to 3D-ShapeNet   (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7298801)
  return {'AUC': AUC, 'MAP': mAP}



if __name__ == '__main__':
  # names = ['a', 'b', 'c']
  # ftrs_per_model = {}
  # for i, n in enumerate(names):
  #   ftrs_per_model.setdefault(n, {}).setdefault(1000, np.random.random((5)))
  #   if i > 1:
  #     ftrs_per_model[n].setdefault(2000, np.random.random((5)))

  # calculate_retrieval_multiscale(ftrs_per_model)
  import os
  base_path = '/home/ran/mesh_walker/runs_compare/0212-12.12.2020..16.23__modelnet_triplet/'
  tst = np.load(os.path.join(base_path, 'features.npz'))
  mids = tst['model_ids']
  labels = tst['labels']
  feats = tst['features']
  # feats, labels, mids = [np.load(os.path.join(base_path, x + '_1k.npy')) for x in ['features', 'labels', 'm_ids']]
  # calculate_all_retrieval_scores(feats, labels, mids)
  # d_mat = Eu_dis_mat_fast(np.mat(feats))
  #
  # rs = convert_rank_gt(labels, d_mat)
  #
  tst = calculate_map_auc(feats, labels, mids)

  # TODO: calculate each one alone

  print(tst)