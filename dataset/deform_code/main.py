import matlab.engine
import h5py
import numpy as np
from py_script import prediction

print("Loading matlab...")
eng = matlab.engine.start_matlab()
print("Matlab loaded!")

#neutral, disgust, surprise, angry, sadness, fear, contempt, happy
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

n_examples = 5
expr = "sadness"
technique = "mode"

pred_vector = prediction.prediction(expr, technique)

indexes = np.random.randint(0, high = len(def_neutral), size = n_examples+1)

for i in range(0,n_examples + 1):

    def_neutral_v = def_neutral[indexes[i]]

    def_v = def_neutral_v + 2*pred_vector

    eng.deform_and_visualize(matlab.double(def_neutral_v.tolist()), matlab.double(def_v.tolist()))

input(...)

