#score = 0.62178
#top 72%

import pandas
import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot


train = pandas.read_csv('train.csv', parse_dates=True)
dt = pandas.DatetimeIndex(train['datetime'])
train['year'] = dt.year
train['month'] = dt.month
train['hour'] = dt.hour
train['weekday'] = dt.weekday

test = pandas.read_csv('test.csv', parse_dates=True)
dt = pandas.DatetimeIndex(test['datetime'])
test['year'] = dt.year
test['month'] = dt.month
test['hour'] = dt.hour
test['weekday'] = dt.weekday

features = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'year', 'month', 'weekday', 'hour']

rf = RandomForestClassifier(n_estimators=100)
rf.fit(train[features], train['count'])
result = rf.predict(test[features])
df = pandas.DataFrame({'datetime':test['datetime'], 'count':result})
df.to_csv('rf_results.csv', index = False, columns=['datetime', 'count'])
