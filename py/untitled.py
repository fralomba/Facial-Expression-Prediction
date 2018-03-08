import h5py
import numpy as np

#Matrix name: def_coeff, labels_expr, labels_id

#neutral, disgust, surprise, angry, sadness, fear, contempt, happy

mat = h5py.File('../dataset/processed_ck.mat')

#Transformation Matrix (neutral, expression, neutral, expression)
def_coeff = np.array(mat["def_coeff"])

#Ids of faces
labels_id = np.array(mat["labels_id"])

#Number of different faces
face_number = np.max(labels_id)

#Loading expression labels
labels_expr = []
with h5py.File('../dataset/processed_ck.mat') as f:
    column = f['labels_expr'][0]
    for row_number in range(len(column)):
        labels_expr.append(''.join(map(chr, f[column[row_number]][:])))

#Expression labels
labels_expr = np.asarray(labels_expr)

#Different types of expression
different_expr = np.sort(np.unique(labels_expr))

#Computing happy matrix
expressions_dict = {}
for expr in different_expr:
    expressions_dict[expr] = def_coeff[np.where(labels_expr == expr)]




