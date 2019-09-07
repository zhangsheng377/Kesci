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
import autosklearn.classification
# import autokeras
import tpot


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
# print(train_data)
# print(train_lable)
# print(test_data)

# 0.9012125403528571 n_estimators=100 0.8827270059637511
# 0.9016471277305751 n_estimators=200 0.8825690280744943
# 0.9032270626044709 n_estimators=500 0.8823715479138569
# clf = AdaBoostClassifier(n_estimators=100)
# scores = cross_val_score(clf, train_data, train_lable, cv=5, verbose=5)
# print(scores.mean())

# svm
# 0.8928785749701833 0.8830430241347959
# clf = SVC()
# scores = cross_val_score(clf, train_data, train_lable, cv=5, verbose=5)
# print(scores.mean())

# 0.8927205970809264
# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
# scores = cross_val_score(clf, train_data, train_lable, cv=5)
# print(scores.mean())

# 0.8830430241347959
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# scores = cross_val_score(clf, train_data, train_lable, cv=5)
# print(scores.mean())

# 0.8755777967598387
# clf = tree.DecisionTreeClassifier()
# scores = cross_val_score(clf, train_data, train_lable, cv=5, verbose=5)
# print(scores.mean())

# 0.8927205970809264
# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
# scores = cross_val_score(clf, train_data, train_lable, cv=5, verbose=5)
# print(scores.mean())

# clf = autosklearn.classification.AutoSklearnClassifier()
# scores = cross_val_score(clf, train_data, train_lable, cv=5, verbose=5)
# print(scores.mean())

# clf = autokeras.classifier()

clf = tpot.TPOTClassifier(verbosity=3, periodic_checkpoint_folder="tpot", warm_start=True)

clf.fit(train_data, train_lable)
test_data, test_ID = handle_data(test_data, test_data_handled_path, need_shuffle=False)
# print(test_data)
# print(test_ID)
predict = clf.predict(test_data)
# print(predict)
result = pd.DataFrame({'ID': test_ID, 'pred': predict})
result.to_csv(result_data_path, index=False)

clf.export('tpot_pipeline.py')

# print(clf.cv_results_)
# print(clf.sprint_statistics())
# print(clf.show_models())

# AdaBoostClassifier(n_estimators=100) need_shuffle=True 0.49890420
# AdaBoostClassifier(n_estimators=100) need_shuffle=False 0.67963529
# AdaBoostClassifier(n_estimators=500) need_shuffle=False 0.54039899
# AdaBoostClassifier(n_estimators=200) need_shuffle=False need_decomposition=True 0.52142756
# AdaBoostClassifier(n_estimators=200) need_shuffle=False need_decomposition=False 0.65858919
# AdaBoostClassifier(n_estimators=150) need_shuffle=False need_decomposition=False 0.65809112
# autosklearn.classification.AutoSklearnClassifier() need_shuffle=False need_decomposition=False 0.68977183
# autosklearn.classification.AutoSklearnClassifier() need_shuffle=False need_decomposition=False 0.71657159
# tpot.TPOTClassifier(verbosity=2) 0.73911188
