import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

rf_r2 = []
rf_mae = []
mlp_r2 = []
mlp_mae = []
knn_r2 = []
knn_mae = []
linear_r2 = []
linear_mae = []
ridge_r2 = []
ridge_mae = []
huber_r2 = []
huber_mae = []
decision_r2 = []
decision_mae = []
svr_r2 = []
svr_mae = []

rf_score_list = []
mlp_score_list = []
knn_score_list = []
linear_score_list = []
ridge_score_list = []
huber_score_list = []
decision_score_list = []
svr_score_list = []

for i in range(1, 17):
    features = pd.read_csv('BS' + str(i) + '.csv', usecols=['Cdistance', 'vitesse', 'Theta', 'NewLoad'])
    labels = pd.read_csv('BS' + str(i) + '.csv', usecols=['lifetime'])
    features = np.array(features)
    labels = np.array(labels)
    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels, test_size=0.2, random_state=15)
    mlp = MLPRegressor(hidden_layer_sizes=(150, 40), max_iter=1000, solver='adam', learning_rate_init=0.01)
    rf = RandomForestRegressor(n_estimators=370)
    knn = KNeighborsRegressor()
    linear = LinearRegression()
    huber = Ridge()
    huber = HuberRegressor(max_iter=500)
    decision = DecisionTreeRegressor()
    svr = SVR()


    #训练
    mlp.fit(train_features, train_labels.ravel())
    rf.fit(train_features, train_labels.ravel())
    knn.fit(train_features, train_labels.ravel())
    linear.fit(train_features, train_labels.ravel())
    # ridge.fit(train_features, train_labels.ravel())
    huber.fit(train_features, train_labels.ravel())
    decision.fit(train_features, train_labels.ravel())
    svr.fit(train_features, train_labels.ravel())
    #预测
    mlp_pre = mlp.predict(test_features)
    rf_pre = rf.predict(test_features)
    knn_pre = rf.predict(test_features)
    linear_pre = rf.predict(test_features)
    ridge_pre = rf.predict(test_features)
    huber_pre = rf.predict(test_features)
    decision_pre = rf.predict(test_features)
    svr_pre = rf.predict(test_features)
    print('mlp:', mlp_pre)
    print('rf:', rf_pre)
    print('knn:', knn_pre)
    print('linear:', linear_pre)
    print('ridge:', ridge_pre)
    print('huber:', huber_pre)
    print('decision:', decision_pre)
    print('svr:', svr_pre)
    #r2
    r2_mlp = r2_score(test_labels, mlp_pre.ravel())
    r2_rf = r2_score(test_labels, rf_pre.ravel())
    r2_knn = r2_score(test_labels, knn_pre.ravel())
    r2_linear = r2_score(test_labels, linear_pre.ravel())
    r2_ridge = r2_score(test_labels, ridge_pre.ravel())
    r2_huber = r2_score(test_labels, huber_pre.ravel())
    r2_decision = r2_score(test_labels, decision_pre.ravel())
    r2_svr = r2_score(test_labels, svr_pre.ravel())
    #mae
    mae_mlp = mean_absolute_error(test_labels, mlp_pre.ravel())
    mae_rf = mean_absolute_error(test_labels, rf_pre.ravel())
    mae_knn = mean_absolute_error(test_labels, knn_pre.ravel())
    mae_linear = mean_absolute_error(test_labels, linear_pre.ravel())
    mae_ridge = mean_absolute_error(test_labels, ridge_pre.ravel())
    mae_huber = mean_absolute_error(test_labels, huber_pre.ravel())
    mae_decision = mean_absolute_error(test_labels, decision_pre.ravel())
    mae_svr = mean_absolute_error(test_labels, svr_pre.ravel())
    # #准确率
    # rf_score = rf.score(test_features, test_labels)
    # mlp_score = mlp.score(test_features, test_labels)
    # knn_score = knn.score(test_features, test_labels)
    # linear_score = linear.score(test_features, test_labels)
    # ridge_score = ridge.score(test_features, test_labels)
    # huber_score = huber.score(test_features, test_labels)
    # decision_score = decision.score(test_features, test_labels)
    # svr_score = svr.score(test_features, test_labels)


    rf_r2.append(r2_rf)
    rf_mae.append(mae_rf)
    mlp_r2.append(r2_mlp)
    mlp_mae.append(mae_mlp)
    # rf_score_list.append(rf_score)
    # mlp_score_list.append(mlp_score)

    knn_r2.append(r2_knn)
    knn_mae.append(mae_knn)
    # knn_score_list.append(knn_score)

    linear_r2.append(r2_knn)
    linear_mae.append(mae_linear)
    # linear_score_list.append(linear_score)

    ridge_r2.append(r2_ridge)
    ridge_mae.append(mae_ridge)
    # ridge_score_list.append(ridge_score)

    huber_r2.append(r2_huber)
    huber_mae.append(mae_huber)
    # huber_score_list.append(huber_score)

    decision_r2.append(r2_decision)
    decision_mae.append(mae_decision)
    # decision_score_list.append(decision_score)

    svr_r2.append(r2_svr)
    svr_mae.append(mae_svr)
    # svr_score_list.append(svr_score)

r2 = pd.DataFrame({'rf': rf_r2, 'mlp': mlp_r2, 'knn': knn_r2, 'linear': linear_r2,
                   'ridge': ridge_r2, 'huber': huber_r2, 'decision': decision_r2, 'svr': svr_r2})

mae = pd.DataFrame({'rf': rf_mae, 'mlp': mlp_mae, 'knn': knn_mae, 'linear': linear_mae,
                   'ridge': ridge_mae, 'huber': huber_mae, 'decision': decision_mae, 'svr': svr_mae})

# score = pd.DataFrame({'rf': rf_score_list, 'mlp': mlp_score_list, 'knn': knn_score_list, 'linear': linear_score_list,
#                    'ridge': ridge_score_list, 'huber': huber_score_list, 'decision': decision_score_list, 'svr': svr_score_list})

r2.to_csv('r2.csv')
mae.to_csv('mae.csv')
# score.to_csv('score.csv')



# features = pd.read_csv('BS1.csv', usecols=['Cdistance', 'vitesse', 'Theta', 'NewLoad'])
# labels = pd.read_csv('BS1.csv', usecols=['lifetime'])
# feature_list = list(features.columns)
# features = np.array(features)
# labels = np.array(labels)
# train_features, test_features, train_labels, test_labels = \
#     train_test_split(features, labels, test_size=0.2, random_state=15)

# cv_score=[]
# n_es = range(100, 1000)
# for i in n_es:
#     rf = RandomForestRegressor(n_estimators=i)
#     scores = cross_val_score(rf, train_features, train_labels.flatten(), cv=10)
#     cv_score.append(scores.mean())
#     print('n_es', i, 'score_mean', scores.mean())
# plt.plot(n_es, cv_score)
# plt.xlabel('es')
# plt.plotting('scores')
# plt.show()

# rf = RandomForestRegressor()
# rf.fit(train_features, train_labels.flatten())
#
# # Use the forest's predict method on the test data
# predictions = rf.predict(test_features)
#
# print(rf.score(train_features, train_labels.flatten()))
# print(rf.score(test_features, test_labels.flatten()))
#
# print(predictions)
# print(test_labels)

# ss_feature = preprocessing.StandardScaler()
# train_features_ss = ss_feature.fit_transform(train_features)
# test_features_ss = ss_feature.fit_transform(test_features)
#
# ss_label = preprocessing.StandardScaler()
# train_labels_ss = ss_label.fit_transform(train_labels)
# test_labels_ss = ss_label.fit_transform(test_labels)
# mlp1 = MLPRegressor(hidden_layer_sizes=(150, 40), max_iter=1000, verbose='True', solver='adam',
#                    early_stopping=True, learning_rate_init=0.01)


# mlp2 = MLPRegressor(hidden_layer_sizes=(150, 40), max_iter=1000, verbose='True', solver='adam',
#                    early_stopping=True, learning_rate_init=0.01)
#
# # mlp1.fit(train_features_ss, train_labels_ss.ravel())
# mlp2.fit(train_features, train_labels.ravel())
# rf = RandomForestRegressor(n_estimators=375)
# rf.fit(train_features, train_labels.ravel())
#
# # pre1 = mlp1.predict(test_features_ss)
# pre2 = mlp2.predict(test_features)
# pre_rf = rf.predict(test_features)
#
# # print("r2处理后", r2_score(test_labels_ss, pre1.ravel()))
# print("r2原始", r2_score(test_labels, pre2.ravel()))
# print("rf_r2", r2_score(test_labels, pre_rf.ravel()))
# # print("mae处理后", mean_absolute_error(test_labels_ss, pre1.ravel()))
# print("mae原始", mean_absolute_error(test_labels, pre2.ravel()))
# print("rf_mae", mean_absolute_error(test_labels, pre_rf.ravel()))