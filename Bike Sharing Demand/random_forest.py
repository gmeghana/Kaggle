# using all features 0.66830
# using only time, workingday, season 0.64388

import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier

train = csv.reader(open('train.csv', 'r'))
header_train = train.next()
train_data = []
for row in train:
	train_data.append(row)
train_data = np.array(train_data)

test = csv.reader(open('test.csv', 'r'))
header_test = test.next()
test_data = []
for row in test:
	test_data.append(row)
test_data = np.array(test_data)

features = ['time', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed']
features = ['time', 'windspeed', 'humidity', 'temp']
train_features = np.zeros([train_data.shape[0], len(features)])
test_features = np.zeros([test_data.shape[0], len(features)])

for key, feature in enumerate(features):
    if feature == 'time':
        train_time = [datetime[11:13] for datetime in train_data[:,0]] 
        train_features[:,key] = np.array(train_time).astype(np.float) 
        test_time = [datetime[11:13] for datetime in test_data[:,0]]
        test_features[:,key] = np.array(test_time).astype(np.float)

    else:
        train_key = header_train.index(feature)
        test_key = header_test.index(feature)

        train_features[:,key] = train_data[:,train_key].astype(np.float)
        test_features[:,key] = test_data[:,test_key].astype(np.float)

train_count = train_data[:,-1].astype(np.float)
test_datetime = test_data[:,0]

rf = RandomForestClassifier(n_estimators = 100)
rf = rf.fit(train_features, train_count)
result = rf.predict(test_features)

count = result.astype(np.int)
f = open('output.csv', 'w')
output_file = csv.writer(f)
output_file.writerow(['datetime', 'count'])

for i in range(len(test_datetime)):
	output_file.writerow([test_datetime[i], count[i]])
f.close()