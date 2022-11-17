from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from random import choice
import numpy as np
import pandas as pd


car = pd.read_csv('BS1.csv', usecols=['Cdistance', 'vitesse', 'Theta', 'NewLoad', 'lifetime'])
car = np.array(car)
ran_car = choice(car)
features = pd.read_csv('BS1.csv', usecols=['Cdistance', 'vitesse', 'Theta', 'NewLoad'])
labels = pd.read_csv('BS1.csv', usecols=['lifetime'])
features = np.array(features)
# print(features)
labels = np.array(labels)
train_features, test_features, train_labels, test_labels = \
    train_test_split(features, labels, test_size=0.2, random_state=15)

br = BaggingRegressor(base_estimator=RandomForestRegressor(), n_estimators=85)
br.fit(train_features, train_labels.ravel())

mlp = MLPRegressor(hidden_layer_sizes=(150, 40), max_iter=1000, solver='adam', learning_rate_init=0.01)
mlp.fit(train_features, train_labels.ravel())

rf = RandomForestRegressor(n_estimators=370)
rf.fit(train_features, train_labels.ravel())

br_pre = []
mlp_pre = []
rf_pre = []
ran = []
for i in range(400):
    br_choice_car = 0
    mlp_choice_car = 0
    rf_choice_car = 0
    true_car = 0
    for j in range(25):
        ran_car = choice(car)
        ran_car_new = np.delete(ran_car, 1)
        ran_car_new = ran_car_new.reshape(1, -1)
        if br.predict(ran_car_new) > 59.23 and ran_car[1] > 59.23:
            br_choice_car += 1
        if mlp.predict(ran_car_new) > 59.23 and ran_car[1] > 59.23:
            mlp_choice_car += 1
        if rf.predict(ran_car_new) > 59.23 and ran_car[1] > 59.23:
            rf_choice_car += 1
        if ran_car[1] > 59.23:
            true_car += 1
    if true_car == 0:
        continue
    br_success = br_choice_car / true_car
    mlp_success = mlp_choice_car / true_car
    rf_success = rf_choice_car / true_car
    ran_success = true_car / 25
    br_pre.append(br_success)
    rf_pre.append(rf_success)
    mlp_pre.append(mlp_success)
    ran.append(ran_success)
    print(br_success, rf_success, mlp_success, ran_success)

trasucc = []
trasucc.append(br_pre)
trasucc.append(rf_pre)
trasucc.append(mlp_pre)
trasucc.append(ran)
test_pretra = pd.DataFrame(data=trasucc)
test_pretra.to_csv('test_pretra.csv', encoding='gbk')

