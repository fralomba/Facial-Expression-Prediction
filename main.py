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
# labels are wrong. Some sure neutral faces are: 49
index = 49
def_neutral_v = def_neutral[index]
# Possible expression: neutral, disgust, surprise, angry, sadness, fear, contempt, happy
expressions = ["disgust", "surprise", "angry", "sadness", "fear", "contempt", "happy", "neutral"]
# Specify your expression
expr = "happy"

# If you want to use mean/median/mode technique, use this method specifying the technique and the expression
# It returns directly the transformation vector to apply to the neutral model

### pred_vector = prediction.m_prediction(expr, tec="mode")

# If you want to use linear/svr/nn regressor, use this method specifying the type of regressor and the expression
# to train the model. It returns a model, then you can use to predict the expression passing it a neutral face
regr = prediction.regressor(expr, tec = "svr", cv_test = False)
print("")

pred_vector = regr.predict (def_neutral_v.reshape(1,-1))

# You can specify an alpha between [1,2] to enhance the expression. The best alpha for angry/disgust/contempt/sadness
# is 1.5; for the others it can be leave to 1. If you use the neural network,
# you might need to raise the alpha up of 0.2
alpha = 1

# Compute and visualize the result
def_v = def_neutral_v + pred_vector * alpha
eng.deform_and_visualize_one(matlab.double(def_neutral_v.tolist()), matlab.double(def_v.tolist()),
                                         expr, "", int(index))

input("Press enter to close...")

'''DA QUI IN POI SI CANCELLA E NON GLI SI DA, SERVE SOLO A NOI PER FARE LE PROVE E SALVARE LE IMMAGINI'''

technique = "mode"
n_examples = 1
alpha = 1
alpha_network = 1.5

#regr_lin = prediction.regressor(expr, "linear")
regr_svr_sig = prediction.regressor(expr, "svr", "sigmoid")
#pred_vector = prediction.m_prediction(expr, technique)

indexes = np.random.randint(0, high = len(def_neutral), size = n_examples+1)

for i in range(0, n_examples):

    indexes[i] = 49

    print("For face with index: ", indexes[i])

    for expr in expressions:
        if expr == 'neutral':
            continue
        elif expr == 'angry' or expr == 'sadness' or expr == 'disgust' or expr == 'contempt':
            alpha = 1.5
            alpha_network = 1.7
        else:
            alpha = 1
            alpha_network = 1.5
        print("Expression: ", expr)

        #Computing m_vector for expr
        pred_vector_mean = prediction.m_prediction(expr, "mean")
        pred_vector_median = prediction.m_prediction(expr, "median")
        pred_vector_mode = prediction.m_prediction(expr, "mode")
        # Computing linear regressor for expr
        regr_lin = prediction.regressor(expr, "linear")
        # Computing svr regressor for expr
        regr_svr_sig = prediction.regressor(expr, "svr", "sigmoid")
        # Computing neural network for expr
        regr_nn = prediction.regressor(expr, "nn")

        #Random neutral model
        def_neutral_v = def_neutral[indexes[i]]

        def_v_mean = def_neutral_v + pred_vector_mean * alpha

        def_v_median = def_neutral_v + pred_vector_median * alpha

        def_v_mode = def_neutral_v + pred_vector_mode * alpha

        def_v3 = def_neutral_v + regr_lin.predict(def_neutral_v.reshape(1, -1)) * alpha

        def_v4 = def_neutral_v + regr_svr_sig.predict(def_neutral_v.reshape(1, -1)) * alpha

        def_v5 = def_neutral_v + regr_nn.predict(def_neutral_v.reshape(1, -1)) * (alpha_network)

        #Visualize and save expression
        eng.deform_and_visualize(matlab.double(def_v_mean.tolist()), matlab.double(def_v_median.tolist()), matlab.double(def_v_mode.tolist()),
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

        if expr == 'angry' or expr == 'sadness' or expr == 'disgust' or expr == 'contempt':
            alpha = 1.5
            alpha_network = 1.7
        else:
            alpha = 1
            alpha_network = 1.5

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

            def_v = def_neutral_v + regr_nn.predict(def_neutral_v.reshape(1, -1)) * alpha_network
            eng.deform_and_visualize_one(matlab.double(def_v.tolist()),
                                         expr, "nn regressor",
                                         "results/" + expr + "/face_" + str(i),
                                         "predicted_sample_" + str(indexes[i]) + "_nn.jpg",
                                         int(indexes[i]))

'''