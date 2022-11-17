import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from random import choice

r2 = []
mae = []

features = pd.read_csv('BS1.csv', usecols=['Cdistance', 'vitesse', 'Theta', 'NewLoad'])
labels = pd.read_csv('BS1.csv', usecols=['lifetime'])
features = np.array(features)
labels = np.array(labels)
train_features, test_features, train_labels, test_labels = \
    train_test_split(features, labels, test_size=0.2, random_state=15)

print(features[0])
print(choice(features))

br = BaggingRegressor(base_estimator=RandomForestRegressor(), n_estimators=85)
br.fit(train_features, train_labels.ravel())

br_pre = br.predict(test_features)

