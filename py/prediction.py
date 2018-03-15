import h5py
import numpy as np
import sys
import scipy.io as sp
from sklearn.cluster import MeanShift, estimate_bandwidth

from sklearn.metrics import mean_squared_error

#Matrix name: def_coeff, labels_expr, labels_id

def prediction(expr = 'happy', technique = 'mean'):
    #neutral, disgust, surprise, angry, sadness, fear, contempt, happy
    mat = h5py.File('../processed_ck.mat')

    #Transformation Matrix (neutral, expression, neutral, expression)
    def_coeff = np.array(mat["def_coeff"])

    #Ids of faces
    labels_id = np.array(mat["labels_id"])

    #Number of different faces
    face_number = np.max(labels_id)

    #Loading expression labels
    labels_expr = []
    with h5py.File('../processed_ck.mat') as f:
        column = f['labels_expr'][0]
        for row_number in range(len(column)):
            labels_expr.append(''.join(map(chr, f[column[row_number]][:])))

    #Expression labels
    labels_expr = np.asarray(labels_expr)

    #Different types of expression
    different_expr = np.sort(np.unique(labels_expr))

    #Computing expressions matrix
    expressions_dict = {}
    for i in range(1, def_coeff.shape[0], 2):

        trans = def_coeff[i,:] - def_coeff[i-1,:]
        trans = np.matrix(trans)
        label = labels_expr[i]
        if label in expressions_dict:
            expressions_dict[label] = np.append(expressions_dict[label], trans, axis=0)
        else:
            expressions_dict[label] = trans


    #Computing means, medians and modes
    mean_dict = {}
    median_dict = {}
    mode_dict = {}

    expressions_dict[expr] = def_coeff[np.where(labels_expr == expr)]

    if technique == 'median':
        # Computing medians
        return np.median(expressions_dict[expr], axis=0)

    elif technique == 'mode':
        # Computing modes
        n_clusters_ = 0
        quantile = 0.3
        while n_clusters_ != 1:
            delta = expressions_dict[expr]

            bandwidth = estimate_bandwidth(delta, quantile=quantile)
            ms = MeanShift(bandwidth=bandwidth)
            ms.fit(delta)
            labels = ms.labels_
            cluster_centers = ms.cluster_centers_

            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)
            quantile += 0.1

        return cluster_centers[0]
    else:
        # Computing means
        return np.mean(expressions_dict[expr], axis=0)


if __name__ == '__main__':
    expr = sys.argv[1]
    technique = sys.argv[2]
    sys.stdout.write(str(prediction(expr, technique)))