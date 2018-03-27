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

#Possible expression: neutral, disgust, surprise, angry, sadness, fear, contempt, happy
expressions = ["disgust", "surprise", "angry", "sadness", "fear", "contempt", "happy", "neutral"]
expr = "angry"
technique = "mode"
n_examples = 5
alpha = 1

#regr_lin = prediction.regressor(expr, "linear")
#regr_svr_sig = prediction.regressor(expr, "svr", "sigmoid")
#pred_vector = prediction.m_prediction(expr, technique)

indexes = np.random.randint(0, high = len(def_neutral), size = n_examples+1)

'''
for i in range(0, n_examples):

    print("For face with index: ", indexes[i])

    for expr in expressions:
        print("Expression: ", expr)

        #Computing m_vector for expr
        pred_vector = prediction.m_prediction(expr, technique)
        # Computing linear regressor for expr
        regr_lin = prediction.regressor(expr, "linear")
        # Computing svr regressor for expr
        regr_svr_sig = prediction.regressor(expr, "svr", "sigmoid")
        # Computing neural network for expr
        regr_nn = prediction.neural_network(expr)

        #Random neutral model
        def_neutral_v = def_neutral[indexes[i]]

        def_v2 = def_neutral_v + pred_vector + alpha

        def_v3 = def_neutral_v + regr_lin.predict(def_neutral_v.reshape(1, -1)) * alpha

        def_v4 = def_neutral_v + regr_svr_sig.predict(def_neutral_v.reshape(1, -1)) * alpha

        def_v5 = def_neutral_v + regr_nn.predict(def_neutral_v.reshape(1, -1)) * (alpha + 0.5)

        #Visualize and save expression
        eng.deform_and_visualize(matlab.double(def_neutral_v.tolist()), matlab.double(def_v2.tolist()),
                                 matlab.double(def_v3.tolist()), matlab.double(def_v4.tolist()), matlab.double(def_v5.tolist()),
                                 expr, technique, "results/" + expr + "_" + technique + "_" + str(i) + ".jpg", int(indexes[i]))


input('Press enter to continue...')
'''

for expr in expressions:

    if expr != 'neutral':
        # Computing m_vector for expr
        pred_vector_mean = prediction.m_prediction(expr, "mean")
        pred_vector_mode = prediction.m_prediction(expr, "mode")
        pred_vector_median = prediction.m_prediction(expr, "median")
        # Computing linear regressor for expr
        regr_lin = prediction.regressor(expr, "linear")
        # Computing svr regressor for expr
        regr_svr_sig = prediction.regressor(expr, "svr", "sigmoid")
        # Computing neural network for expr
        regr_nn = prediction.neural_network(expr)

    for i in range(0, n_examples):

        def_neutral_v = def_neutral[indexes[i]]

        if expr == 'neutral':
            eng.deform_and_visualize_one(matlab.double(def_neutral_v.tolist()),
                                         "neutral", "",
                                         "results/neutral/face_" + str(i), "predicted_sample_" + str(indexes[i]) + ".jpg",
                                         int(indexes[i]))
        else:

            def_v = def_neutral_v + pred_vector_mean * alpha
            eng.deform_and_visualize_one(matlab.double(def_v.tolist()),
                                         expr, "mean",
                                         "results/" + expr + "/face_" + str(i),
                                         "predicted_sample_" + str(indexes[i]) + "_mean.jpg",
                                         int(indexes[i]))

            def_v = def_neutral_v + pred_vector_median * alpha
            eng.deform_and_visualize_one(matlab.double(def_v.tolist()),
                                         expr, "median",
                                         "results/" + expr + "/face_" + str(i),
                                         "predicted_sample_" + str(indexes[i]) + "_median.jpg",
                                         int(indexes[i]))

            def_v = def_neutral_v + pred_vector_mode * alpha
            eng.deform_and_visualize_one(matlab.double(def_v.tolist()),
                                         expr, "mode",
                                         "results/" + expr + "/face_" + str(i),
                                         "predicted_sample_" + str(indexes[i]) + "_mode.jpg",
                                         int(indexes[i]))

            def_v = def_neutral_v + regr_lin.predict(def_neutral_v.reshape(1, -1)) * alpha
            eng.deform_and_visualize_one(matlab.double(def_v.tolist()),
                                         expr, "linear regressor",
                                         "results/" + expr + "/face_" + str(i),
                                         "predicted_sample_" + str(indexes[i]) + "_lin.jpg",
                                         int(indexes[i]))

            def_v = def_neutral_v + regr_svr_sig.predict(def_neutral_v.reshape(1, -1)) * alpha
            eng.deform_and_visualize_one(matlab.double(def_v.tolist()),
                                         expr, "svr regressor",
                                         "results/" + expr + "/face_" + str(i),
                                         "predicted_sample_" + str(indexes[i]) + "_svr.jpg",
                                         int(indexes[i]))

            def_v = def_neutral_v + regr_nn.predict(def_neutral_v.reshape(1, -1)) * alpha
            eng.deform_and_visualize_one(matlab.double(def_v.tolist()),
                                         expr, "nn regressor",
                                         "results/" + expr + "/face_" + str(i),
                                         "predicted_sample_" + str(indexes[i]) + "_nn.jpg",
                                         int(indexes[i]))

