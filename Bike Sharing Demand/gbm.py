# score = 0.41691
# top 12%

import pandas
import datetime
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
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
for col in ['casual', 'registered', 'count']:
    train[col + '_log'] = train[col].apply(lambda x: np.log1p(x))

'''
dt = pandas.DatetimeIndex(train['datetime'])
train_set = train[dt.day <= 16]
validation_set = train[dt.day > 16]

hyperparameters = {'min_samples_leaf':[10,20,30], 'max_depth':[5,10,15], 'learning_rate':[0.1, 0.05, 0.01]}
gbm = GradientBoostingRegressor(n_estimators=500)
gs = GridSearchCV(gbm, hyperparameters).fit(train_set[features], train_set['count_log'])
gs.best_params_

# best params: learning rate = 0.1, max_depth = 5, min_samples_leaf = 30

train_error = []
validation_error = []
x = range(1,500,10)
for i in x:
	gbm = GradientBoostingRegressor(n_estimators=i, learning_rate=0.1, max_depth=5, min_samples_leaf=30)
	gbm.fit(train_set[features], train_set['count_log'])
	result = gbm.predict(train_set[features])
	train_error.append(mean_absolute_error(result, train_set['count_log']))
	result = gbm.predict(validation_set[features])
	validation_error.append(mean_absolute_error(result, validation_set['count_log']))

pyplot.style.use('ggplot')
pyplot.plot(x, train_error)
pyplot.plot(x, validation_error)
pyplot.show()

# best n_estimators = 85
'''
'''
#score = 0.42122
gbm_casual = GradientBoostingRegressor(n_estimators=85, learning_rate=0.1, max_depth=5, min_samples_leaf=30)
gbm_registered = GradientBoostingRegressor(n_estimators=85, learning_rate=0.1, max_depth=5, min_samples_leaf=30)

gbm_casual.fit(train[features].values, train['casual_log'].values)
gbm_registered.fit(train[features].values, train['registered_log'].values)

result = np.expm1(gbm_casual.predict(test[features])) + np.expm1(gbm_registered.predict(test[features]))
'''
gbm = GradientBoostingRegressor(n_estimators=85, learning_rate=0.1, max_depth=5, min_samples_leaf=30)
gbm.fit(train[features], train['count_log'])
result = np.expm1(gbm.predict(test[features]))

df = pandas.DataFrame({'datetime':test['datetime'], 'count':result})
df.to_csv('gbm_results.csv', index = False, columns=['datetime', 'count'])