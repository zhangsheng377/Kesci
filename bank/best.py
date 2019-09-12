import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MinMaxScaler
import pathlib
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn import tree
from sklearn.decomposition import PCA
# import autosklearn.classification
# import autokeras
import tpot
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier


def handle_data(data, data_handled_path, need_shuffle=False, need_decomposition=False):
    lable = None
    if 'y' in data.columns:
        # 取出训练数据的标签
        lable = data.y
        data.drop(labels="y", axis=1, inplace=True)
    else:
        lable = data.ID
    # 去除训练数据的ID
    data.drop(labels="ID", axis=1, inplace=True)
    # 对月份不需要独热编码
    month_mapping = {'jan': 1., 'feb': 2., 'mar': 3., 'apr': 4., 'may': 5., 'jun': 6.,
                     'jul': 7., 'aug': 8., 'sep': 9., 'oct': 10., 'nov': 11., 'dec': 12.,
                     }
    data.month = data.month.map(month_mapping)
    # 对训练数据进行独热编码
    data = pd.get_dummies(data)
    # 归一化
    keys = data.keys()
    data = minmax_scale(data)
    data = pd.DataFrame(data, columns=keys)
    # 乱序
    if need_shuffle:
        data = shuffle(data)
    # 降维
    if need_decomposition:
        keys = data.keys()
        pca = PCA()
        pca.fit(data)
        data = pca.transform(data)
        data = pd.DataFrame(data, columns=keys)

    if data_handled_path is not None:
        data.to_csv(data_handled_path)

    return data, lable


path = pathlib.Path.cwd()
data_path = path.joinpath("data")
train_data_path = data_path.joinpath("train_set.csv")
train_data = pd.read_csv(train_data_path)
test_data_path = data_path.joinpath("test_set.csv")
test_data = pd.read_csv(test_data_path)
train_data_handled_path = data_path.joinpath("train_handled_set.csv")
test_data_handled_path = data_path.joinpath("test_handled_set.csv")
result_path = path.joinpath("result")
result_data_path = result_path.joinpath("predict_set.csv")

train_data, train_lable = handle_data(train_data, train_data_handled_path, need_shuffle=False, need_decomposition=False)

clf = make_pipeline(
    StackingEstimator(estimator=BernoulliNB(alpha=10.0, fit_prior=True)),
    XGBClassifier(learning_rate=0.1, max_depth=6, min_child_weight=19, n_estimators=100, nthread=1,
                  subsample=0.9500000000000001)
)

clf.fit(train_data, train_lable)
test_data, test_ID = handle_data(test_data, test_data_handled_path, need_shuffle=False)
# print(test_data)
# print(test_ID)
predict = clf.predict(test_data)
# print(predict)
result = pd.DataFrame({'ID': test_ID, 'pred': predict})
result.to_csv(result_data_path, index=False)
