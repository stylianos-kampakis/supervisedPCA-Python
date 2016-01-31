from SupervisedPCA import SupervisedPCARegressor
from SupervisedPCA import SupervisedPCAClassifier
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn import datasets
import numpy as np

diabetes=datasets.load_iris()
X = diabetes.data
Y = diabetes.target

spca = SupervisedPCAClassifier()
spca.fit(X, Y,threshold=1.7)
print(spca._model.coef_)