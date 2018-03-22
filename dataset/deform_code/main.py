import matlab.engine
import h5py
import numpy as np
from py_script import prediction


expr = "happy"
regr_svr_poly = prediction.regressor(expr, "svr", "rbf")


input()
print("Loading matlab...")
eng = matlab.engine.start_matlab()
print("Matlab loaded!")

mat = h5py.File('../processed_ck.mat')

#Transformation Matrix (neutral, expression, neutral, expression)
def_coeff = np.array(mat["def_coeff"])
labels_expr = []

with h5py.File('../processed_ck.mat') as f:
    column = f['labels_expr'][0]
    for row_number in range(len(column)):
        labels_expr.append(''.join(map(chr, f[column[row_number]][:])))

labels_expr = np.asarray(labels_expr)

def_neutral = def_coeff[labels_expr == 'neutral']

#disgust, surprise, angry, sadness, fear, contempt, happy
expr = "sadness"
technique = "mean"
n_examples = 5
alpha = 1.5
kernels = ["poly", "rbf", "sigmoid"]

#regr_lin = prediction.regressor(expr, "linear")
#regr_svr_rbf = prediction.regressor(expr, "svr", "rbf")
regr_svr_poly = prediction.regressor(expr, "svr", "poly")
#regr_svr_sig = prediction.regressor(expr, "svr", "sigmoid")
pred_vector = prediction.m_prediction(expr, technique)

indexes = np.random.randint(0, high = len(def_neutral), size = n_examples+1)

for i in range(0, n_examples):

    def_neutral_v = def_neutral[indexes[i]]

    def_v2 = def_neutral_v + regr_svr_poly.predict(def_neutral_v.reshape(1, -1)) * alpha

    def_v3 = def_neutral_v + regr_svr_rbf.predict(def_neutral_v.reshape(1, -1)) * alpha

    def_v4 = def_neutral_v + regr_svr_sig.predict(def_neutral_v.reshape(1, -1)) * alpha


    eng.deform_and_visualize(matlab.double(def_neutral_v.tolist()), matlab.double(def_v2.tolist()),
                             matlab.double(def_v3.tolist()), matlab.double(def_v4.tolist()),
                             expr, technique, "results/" + expr + "_" + technique + "_" + str(i) + ".jpg", int(indexes[i]))


input('Press enter to continue...')

'''
pred_vector = prediction.m_prediction(expr, technique)

indexes = np.random.randint(0, high = len(def_neutral), size = n_examples+1)

for i in range(0,n_examples):

    def_neutral_v = def_neutral[indexes[i]]

    def_v = def_neutral_v + pred_vector*alpha

    eng.deform_and_visualize(matlab.double(def_neutral_v.tolist()), matlab.double(def_v.tolist()),
                             expr, technique, "results/" + expr + "_" + technique + "_" + str(i) + ".jpg")

input('Press enter to continue...')
'''