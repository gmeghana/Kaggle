# Recursive Feature Elimination
'''
The Recursive Feature Elimination (RFE) method is a feature selection 
approach. It works by recursively removing attributes and building a 
model on those attributes that remain. It uses the model accuracy to 
identify which attributes (and combination of attributes) contribute 
the most to predicting the target attribute.
'''


import numpy as np
import csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


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

# create a base classifier used to evaluate a subset of attributes
model = LogisticRegression()

# create the RFE model and select 3 attributes
rfe = RFE(model, 3)
rfe = rfe.fit(train_features, train_count)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)

#[False  True False  True  True False False False False]
#[6 1 2 1 1 5 4 7 3]
#top three features: season, workingday, weather

