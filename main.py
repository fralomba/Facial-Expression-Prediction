import matlab.engine
import h5py
import numpy as np
from py_script import prediction

print("Starting matlab engine...")
eng = matlab.engine.start_matlab()
print("Matlab loaded!")

print("Loading neutral faces...")
mat = h5py.File('data/processed_ck.mat')

def_coeff = np.array(mat["def_coeff"])
labels_expr = []

with h5py.File('data/processed_ck.mat') as f:
    column = f['labels_expr'][0]
    for row_number in range(len(column)):
        labels_expr.append(''.join(map(chr, f[column[row_number]][:])))

labels_expr = np.asarray(labels_expr)

different_expr = np.sort(np.unique(labels_expr))

def_neutral = def_coeff[labels_expr == 'neutral']

# Index of a neutral face of the dataset. The range is 0 - 326; It isn't sure that is a real neutral face, since some
# labels are wrong. Some sure neutral faces are: 49, 242, 159, 35, 66, 16, 104, 266
index = 49
def_neutral_v = def_neutral[index]
# Possible expression: neutral, disgust, surprise, angry, sadness, fear, contempt, happy
expressions = ["disgust", "surprise", "angry", "sadness", "fear", "contempt", "happy", "neutral"]
# Specify your expression
expr = "happy"

# If you want to use mean/median/mode technique, use this method specifying the technique and the expression
# It returns directly the transformation vector to apply to the neutral model

###pred_vector = prediction.m_prediction(expr, tec="mode")

# If you want to use linear/svr/nn regressor, use this method specifying the type of regressor and the expression
# to train the model. It returns a model, then you can use to predict the expression passing it a neutral face
regr = prediction.regressor(expr, tec = "svr")
pred_vector = regr.predict (def_neutral_v.reshape(1,-1))

# You can specify an alpha between [1,2] to enhance the expression. The best alpha for angry/disgust/contempt/sadness
# is 1.5; for the others it can be leave to 1. If you use the neural network,
# you might need to raise the alpha up of 0.2
alpha = 1

# Compute and visualize the result
def_v = def_neutral_v + pred_vector * alpha
eng.deform_and_visualize(matlab.double(def_neutral_v.tolist()), matlab.double(def_v.tolist()),
                                         expr, "", int(index))

input("Press enter to close..")
