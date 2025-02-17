import numpy as np
from sklearn.utils import shuffle
# load training data
training_data = np.load('data/training_data.npy') 
prices = np.load('data/prices.npy')

training_data, prices = shuffle(training_data, prices, random_state=0)
'''The first 4 samples are:

  [[2.0150e+03 4.1000e+04 1.9670e+01 1.5820e+03 1.2620e+02 5.0000e+00
  1.0000e+00 0.0000e+00 1.0000e+00 0.0000e+00 0.0000e+00 0.0000e+00
  1.0000e+00 0.0000e+00]
 [2.0110e+03 4.6000e+04 1.8200e+01 1.1990e+03 8.8700e+01 5.0000e+00
  1.0000e+00 0.0000e+00 0.0000e+00 1.0000e+00 0.0000e+00 0.0000e+00
  1.0000e+00 0.0000e+00]
 [2.0120e+03 8.7000e+04 2.0770e+01 1.2480e+03 8.8760e+01 7.0000e+00
  1.0000e+00 0.0000e+00 1.0000e+00 0.0000e+00 0.0000e+00 0.0000e+00
  1.0000e+00 0.0000e+00]
 [2.0130e+03 8.6999e+04 2.3080e+01 1.4610e+03 6.3100e+01 5.0000e+00
  1.0000e+00 0.0000e+00 1.0000e+00 0.0000e+00 0.0000e+00 0.0000e+00
  1.0000e+00 0.0000e+00]]
The first 4 prices are:
  [12.5  4.5  6.   3.5]'''

import sklearn.preprocessing as sk
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
def normalize(train_data, test_data):
    scaler = sk.StandardScaler()
    scaler.fit(train_data)
    scaled_train_data = scaler.transform(train_data)
    scaled_test_data = scaler.transform(test_data)
    
    return scaled_train_data, scaled_test_data
num_examples_fold = len(prices) // 3
train_1, labels_1 = training_data[:num_examples_fold], prices[:num_examples_fold]
train_2, labels_2 = training_data[num_examples_fold:2*num_examples_fold], prices[num_examples_fold:2*num_examples_fold]
train_3, labels_3 = training_data[num_examples_fold*2:], prices[num_examples_fold*2:]
print(train_1.shape, train_2.shape, train_3.shape)

#(1626, 14) (1626, 14) (1627, 14)
def normalize_train_and_eval(model, train_data, train_labels, test_data, test_labels):
    scaled_train_data, scaled_test_data = normalize(train_data, test_data)
    model.fit(scaled_train_data, train_labels)
    predicted_prices = model.predict(scaled_test_data)
    return mean_squared_error(test_labels, predicted_prices), mean_absolute_error(test_labels, predicted_prices)
    
linear_model = LinearRegression()
mse_1, mae_1 = normalize_train_and_eval(linear_model, train_data=np.concatenate((train_1, train_2)), train_labels=np.concatenate((labels_1, labels_2)), test_data=train_3, test_labels=labels_3)

linear_model = LinearRegression()
mse_2, mae_2 = normalize_train_and_eval(linear_model, train_data=np.concatenate((train_1, train_3)), train_labels=np.concatenate((labels_1, labels_3)), test_data=train_2, test_labels=labels_2)


linear_model = LinearRegression()
mse_3, mae_3 = normalize_train_and_eval(linear_model, train_data=np.concatenate((train_2, train_3)), train_labels=np.concatenate((labels_2, labels_3)), test_data=train_1, test_labels=labels_1)
mae = (mae_1 + mae_2 + mae_3) / 3
mse = (mse_1 + mse_2 + mse_3) / 3
print(mae, mse)
#1.3195985158284504 3.1674890518188477
print(prices.min(), prices.max())
#0.44 16.0
