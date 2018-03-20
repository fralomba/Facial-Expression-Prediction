import h5py
import numpy as np
import sys
from sklearn.cluster import MeanShift, estimate_bandwidth

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def load_data():
    # neutral, disgust, surprise, angry, sadness, fear, contempt, happy
    mat = h5py.File('../processed_ck.mat')

    # Transformation Matrix (neutral, expression, neutral, expression)
    def_coeff = np.array(mat["def_coeff"])

    # Loading expression labels
    labels_expr = []
    with h5py.File('../processed_ck.mat') as f:
        column = f['labels_expr'][0]
        for row_number in range(len(column)):
            labels_expr.append(''.join(map(chr, f[column[row_number]][:])))

    # Expression labels
    labels_expr = np.asarray(labels_expr)

    # Computing expressions matrix
    expressions_dict = {}
    for i in range(1, def_coeff.shape[0], 2):

        trans = def_coeff[i, :] - def_coeff[i - 1, :]
        trans = np.matrix(trans)
        label = labels_expr[i]
        if label in expressions_dict:
            expressions_dict[label] = np.append(expressions_dict[label], trans, axis=0)
        else:
            expressions_dict[label] = trans

    return expressions_dict

def prediction(expr = 'happy', technique = 'mean', bandwidth=None):

    expressions_dict = load_data()

    #expressions_dict[expr] = def_coeff[np.where(labels_expr == expr)]

    if technique == 'median':
        # Computing medians
        return np.median(expressions_dict[expr], axis=0)

    elif technique == 'mode':
        if bandwidth != None:
            delta = expressions_dict[expr]

            ms = MeanShift(bandwidth=bandwidth)
            ms.fit(delta)
            labels = ms.labels_
            cluster_centers = ms.cluster_centers_

            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)
            return cluster_centers

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

'''
def linearRegressor():
    # neutral, disgust, surprise, angry, sadness, fear, contempt, happy
    mat = h5py.File('../processed_ck.mat')

    # Transformation Matrix (neutral, expression, neutral, expression)
    def_coeff = np.array(mat["def_coeff"])

    # Ids of faces
    labels_id = np.array(mat["labels_id"])

    # Number of different faces
    face_number = np.max(labels_id)

    # Loading expression labels
    labels_expr = []
    with h5py.File('../processed_ck.mat') as f:
        column = f['labels_expr'][0]
        for row_number in range(len(column)):
            labels_expr.append(''.join(map(chr, f[column[row_number]][:])))

    # Expression labels
    labels_expr = np.asarray(labels_expr)

    # Different types of expression
    different_expr = np.sort(np.unique(labels_expr))

    # Computing expressions matrix
    expressions_dict = {}
    for i in range(1, def_coeff.shape[0], 2):

        trans = def_coeff[i, :] - def_coeff[i - 1, :]
        trans = np.matrix(trans)
        label = labels_expr[i]
        if label in expressions_dict:
            expressions_dict[label] = np.append(expressions_dict[label], trans, axis=0)
        else:
            expressions_dict[label] = trans

    expressions_dict[expr] = def_coeff[np.where(labels_expr == expr)]

    set = expressions_dict['neutral']
    np.shuffle(set)
    
    realExpr = expressions_dict['happy']

    #input = input[0:69]
    Xtrain,Ytrain,xtest,ytest = train_test_split(set,realExpr,test_size=0.25,random_state=4)

    regr = LinearRegression()
    regr.fit(input, realExpr)

    pred = regr.predict(input[0].reshape(1, -1))
    print(pred)
    print(realExpr[0])

#------------------altro modo che sono sicuro che vada bene ma non riesco a capire cosa sono le y, cioè a parole dobbiamo
#                  dividere in X e y i vettori neutrali (X) e quelli delle espressioni(y) e poi dovrebbe andare questo codice qui sotto.

    # Create training and test sets
    X_train, X_test, y_train, y_test = train_test_split(set, y, test_size=0.25, random_state=42)

    # Create the regressor: reg_all
    reg_all = LinearRegression()

    # Fit the regressor to the training data
    reg_all.fit(X_train, y_train)

    # Predict on the test data: y_pred
    y_pred = reg_all.predict(X_test)

    # Compute and print R^2 and RMSE
    print("R^2: {}".format(reg_all.score(X_test, y_test)))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Root Mean Squared Error: {}".format(rmse))
'''

if __name__ == '__main__':

    #Mean Shift Study

    mat = h5py.File('../data/processed_ck.mat')

    # Transformation Matrix (neutral, expression, neutral, expression)
    def_coeff = np.array(mat["def_coeff"])

    # Ids of faces
    labels_id = np.array(mat["labels_id"])

    # Number of different faces
    face_number = np.max(labels_id)

    # Loading expression labels
    labels_expr = []
    with h5py.File('../data/processed_ck.mat') as f:
        column = f['labels_expr'][0]
        for row_number in range(len(column)):
            labels_expr.append(''.join(map(chr, f[column[row_number]][:])))

    # Expression labels
    labels_expr = np.asarray(labels_expr)

    # Different types of expression
    different_expr = np.sort(np.unique(labels_expr))

    # Computing expressions matrix
    expressions_dict = {}
    for i in range(1, def_coeff.shape[0], 2):

        trans = def_coeff[i, :] - def_coeff[i - 1, :]
        trans = np.matrix(trans)
        label = labels_expr[i]
        if label in expressions_dict:
            expressions_dict[label] = np.append(expressions_dict[label], trans, axis=0)
        else:
            expressions_dict[label] = trans

    for expr in different_expr:
        # Computing modes
        n_clusters_ = 0
        quantile = 0.3
        if expr == 'neutral':
            continue
        print("For expression ", expr)

        while quantile < 1:

            delta = expressions_dict[expr]

            bandwidth = estimate_bandwidth(delta, quantile=quantile)
            ms = MeanShift(bandwidth=bandwidth)
            ms.fit(delta)
            labels = ms.labels_
            cluster_centers = ms.cluster_centers_

            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)
            quantile += 0.1
            print(bandwidth, " = ", n_clusters_)



    from sklearn.svm import SVR
    from sklearn.multioutput import MultiOutputRegressor

    clf = MultiOutputRegressor(SVR(C=1.0, epsilon=0.2))

    y = expressions_dict['happy']
    x = def_coeff[np.where(labels_expr == 'neutral')]
    x = x[0:69]

    clf.fit(x,y)

    print(clf.predict(x[0].reshape(1, -1)))
    print(y[0])

