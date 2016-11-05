import numpy as np
import csv
from sklearn import metrics
from sklearn.linear_model import Lasso

train = csv.reader(open('train.csv', 'r'))
header_train = train.__next__()
train_data = []
for row in train:
	train_data.append(row)
train_data = np.array(train_data)

features = ['time', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed']
train_features = np.zeros([train_data.shape[0], len(features)])

for key, feature in enumerate(features):
    if feature == 'time':
        train_time = [datetime[11:13] for datetime in train_data[:,0]] 
        train_features[:,key] = np.array(train_time).astype(np.float) 
    else:
        train_key = header_train.index(feature)
        train_features[:,key] = train_data[:,train_key].astype(np.float)

train_count = train_data[:,-1].astype(np.float)

print(train_features)
print(train_count)

# fit an Extra Trees model to the data
lasso = Lasso(alpha=.05)
lasso.fit(train_features, train_count)

# display the relative importance of each attribute
print(lasso.coef_)

# larger score means more importance
# [  7.57855033  21.5519792   -6.77452288  -0. -2.72768202  2.28674471   4.4277883   -2.23714013   0.36618516]
# time, season, temp, atemp