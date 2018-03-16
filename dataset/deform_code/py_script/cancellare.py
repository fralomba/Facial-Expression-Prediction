




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
