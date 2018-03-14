import h5py
import numpy as np
import scipy.io as sp
from sklearn.cluster import MeanShift, estimate_bandwidth

from sklearn.metrics import mean_squared_error

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

for expr in different_expr:
    if expr == 'neutral':
        continue

    expressions_dict[expr] = def_coeff[np.where(labels_expr == expr)]

    # Computing means
    mean_dict[expr] = np.mean(expressions_dict[expr], axis=0)

    #Computing medians
    median_dict[expr] = np.median(expressions_dict[expr], axis=0)

    #Computing modes
    n_clusters_ = 0
    quantile = 0.3
    while n_clusters_ != 1 :
        delta = expressions_dict[expr]

        bandwidth = estimate_bandwidth(delta, quantile=quantile)
        ms = MeanShift(bandwidth=bandwidth)
        ms.fit(delta)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        quantile += 0.1

    mode_dict[expr] = cluster_centers[0]

#save matlab matrix for each expression
sp.savemat('../dataset/mean_coefficients/happy.mat',{'mean_happy':mean_dict['happy']})
sp.savemat('../dataset/mean_coefficients/disgust.mat',{'mean_disgust':mean_dict['disgust']})
sp.savemat('../dataset/mean_coefficients/surprise.mat',{'mean_surprise':mean_dict['surprise']})
sp.savemat('../dataset/mean_coefficients/angry.mat',{'mean_angry':mean_dict['angry']})
sp.savemat('../dataset/mean_coefficients/sadness.mat',{'mean_sadness':mean_dict['sadness']})
sp.savemat('../dataset/mean_coefficients/fear.mat',{'mean_fear':mean_dict['fear']})
sp.savemat('../dataset/mean_coefficients/contempt.mat',{'mean_contempt':mean_dict['contempt']})

'''
Secondo me questo non ci serve più!

#Computing numbers of mode
for expr in expressions_dict:

    quantile = 0.3
    n_clusters_ = 0
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
'''
'''
Calcolo dell'errore quadratico medio medio, la media viene l'ottimo, ma il Pala ha detto che non ha senso
errors_dict = {}
for expr in expressions_dict:

    matrix_errors = np.zeros((expressions_dict[expr].shape[0], 3))

    if expr == 'neutral':
        continue

    for i in range(expressions_dict[expr].shape[0]):
        matrix_errors[i][0] = (mean_squared_error(expressions_dict[expr][i], mean_dict[expr]))
        matrix_errors[i][1] = (mean_squared_error(expressions_dict[expr][i], median_dict[expr]))
        matrix_errors[i][2] = (mean_squared_error(expressions_dict[expr][i], mode_dict[expr]))

    errors_dict[expr] = matrix_errors.mean(0)

    print('Errors ' + expr)
    print('     mean = ', errors_dict[expr][0], ', median = ', errors_dict[expr][1], ', mode = ', errors_dict[expr][2])
    print(' ')
'''

'''
from sklearn import linear_model

input = expressions_dict['neutral']
output = expressions_dict['happy']

input = input[0:69]

regr = linear_model.LinearRegression()
regr.fit(input, output)

pred = regr.predict(input[0].reshape(1,-1))
print(pred)
print(output[0])



from sklearn.svm import SVC
output = expressions_dict['happy']
input = expressions_dict['neutral']

input = input[0:69]

clf = SVC()
clf.fit(input, output)

'''