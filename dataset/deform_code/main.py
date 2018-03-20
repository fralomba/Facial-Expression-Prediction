import matlab.engine
import h5py
import numpy as np
from py_script import prediction

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
expr = "disgust"
technique = "mode"
n_examples = 5
alpha = 1

pred_vector = prediction.prediction(expr, technique)

indexes = np.random.randint(0, high = len(def_neutral), size = n_examples+1)

for i in range(0,n_examples):

    def_neutral_v = def_neutral[indexes[i]]

    def_v = def_neutral_v + pred_vector*alpha

    eng.deform_and_visualize(matlab.double(def_neutral_v.tolist()), matlab.double(def_v.tolist()),
                             expr, technique, "results/" + expr + "_" + technique + "_" + str(i) + ".jpg")

input('Press enter to continue...')

techniques = ["mean", "median", "mode"]

for tec in techniques:
    pred_vector = prediction.prediction(expr, tec)

    def_neutral_v = def_neutral[indexes[0]]

    def_v = def_neutral_v + pred_vector

    eng.deform_and_visualize(matlab.double(def_neutral_v.tolist()), matlab.double(def_v.tolist()),
                             expr, tec, "results/" + expr + "_" + tec + ".jpg")

input('Press enter to continue...')