import tensorflow as tf
import random as r
import numpy as np
import matplotlib.pyplot as plt

"""
Modified from Derek Chia's code
https://towardsdatascience.com/a-line-by-line-laymans-guide-to-linear-regression-using-tensorflow-3c0392aa9e1f
"""

class SimpleLinearRegression:
  def __init__(self, initializer=None):
    self.var = initializer
    if initializer==None:
      self.var = tf.random.uniform(shape=[], minval=0., maxval=1.)

    self.m = tf.Variable(1., shape=tf.TensorShape(None))
    self.b = tf.Variable(self.var,dtype='float')

  def predict(self, x):
    return tf.reduce_sum(self.m * x, 1) + self.b

  def mse(self, true, predicted):
    print(true,predicted,tf.reduce_mean(tf.square(true-predicted)))
    return tf.reduce_mean(tf.square(true-predicted))

  def update(self, X, y, learning_rate):
    with tf.GradientTape(persistent=True) as g:
      loss = self.mse(y, self.predict(X))

    print("Loss: ", loss)

    dy_dm = g.gradient(loss, self.m)
    dy_db = g.gradient(loss, self.b)

    self.m.assign_sub(learning_rate * dy_dm)
    self.b.assign_sub(learning_rate * dy_db)

  def train(self, X, y, learning_rate=0.01, epochs=5):


    if len(X.shape)==1:
      X=tf.reshape(X,[X.shape[0],1])

    self.m.assign([self.var]*X.shape[-1])

    for i in range(epochs):
      print("Epoch: ", i)
      self.update(X, y, learning_rate)

for i in range(100):
  x_train = i
  y_train = 2*i + r.random()

x_train = np.array([x_train])
y_train = np.array([y_train])

for j in range(3):
  for i in range(100):
    x_test = i
    y_test = 2*i + r.random()

x_test = np.array([x_test])
y_test = np.array([y_test])

if __name__ == "__main__":

  if True:
    x_train = np.genfromtxt("C:/Users/cdkte/Downloads/line_data/641_day4.csv",delimiter=",",dtype="float")
    x_train = x_train[:,6:]
    print("NAN TEST1",np.sum(x_train))

    y_train = np.empty((x_train.shape[0],),dtype=float)
    y_train.fill(4.)
    print(y_train.shape)
  if True:
    x = np.genfromtxt("C:/Users/cdkte/Downloads/line_data/641_day6.csv",delimiter=",",dtype="float")
    x = x[:,6:]
    count = 0
    for i in range(len(x)):
      if not np.isnan(np.sum(x[i])):
        x_train = np.concatenate((x_train,[x[i]]),axis=0)
      else:
        count += 1
    print("NAN TEST",np.sum(x))
    y = np.empty((x.shape[0]-count,),dtype=float)
    y.fill(6.)
    y_train = np.concatenate((y_train,y))
  if True:
    x = np.genfromtxt("C:/Users/cdkte/Downloads/line_data/641_day8.csv",delimiter=",",dtype="float")
    x = x[:,6:]
    count = 0
    for i in range(len(x)):
      if not np.isnan(np.sum(x[i])):
        x_train = np.concatenate((x_train,[x[i]]),axis=0)
      else:
        count += 1
    print("NAN TEST",np.sum(x))
    y = np.empty((x.shape[0]-count,),dtype=float)
    y.fill(8.)
    y_train = np.concatenate((y_train,y))

  if True:
    x = np.genfromtxt("C:/Users/cdkte/Downloads/line_data/641_day10.csv",delimiter=",",dtype="float")
    x = x[:,6:]
    count = 0
    for i in range(len(x)):
      if not np.isnan(np.sum(x[i])):
        x_train = np.concatenate((x_train,[x[i]]),axis=0)
      else:
        count += 1
    print("NAN TEST",np.sum(x))
    y = np.empty((x.shape[0]-count,),dtype=float)
    y.fill(10.)
    y_train = np.concatenate((y_train,y))
  if True:
    x = np.genfromtxt("C:/Users/cdkte/Downloads/line_data/641_day12.csv",delimiter=",",dtype="float")
    x = x[:,6:]
    count = 0
    for i in range(len(x)):
      if not np.isnan(np.sum(x[i])):
        x_train = np.concatenate((x_train,[x[i]]),axis=0)
      else:
        count += 1
    print("NAN TEST",np.sum(x))
    y = np.empty((x.shape[0]-count,),dtype=float)
    y.fill(12.)
    y_train = np.concatenate((y_train,y))


  x_test = x_train
  y_test = y_train
  print(x_train.shape,y_train.shape)

  from keras.datasets import boston_housing
  #(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

  print(x_train.shape,y_train.shape)
  if np.nan in x_train or np.inf in x_train or -np.inf in x_train:
    print("test")

  mean_label = y_train.mean(axis=0)
  std_label = y_train.std(axis=0)

  mean_feat = x_train.mean(axis=0)
  mean_feat = np.nanmean(x_train,axis=0)
  for i in range(x_train.shape[0]):
    if np.inf in x_train[i] or -np.inf in x_train[i] or np.nan in x_train[i] or 0 in x_train[i]:
      print("ns",x_train[i])
      x_train.remove(i)
  std_feat = np.nanstd(x_train,axis=0)

  x_train_norm = (x_train-mean_feat)/std_feat
  y_train_norm = (y_train-mean_label)/std_label
  linear_model = SimpleLinearRegression(0)
  linear_model.train(x_train_norm, y_train_norm, learning_rate=0.1, epochs=50)

  # standardize
  x_test = (x_test-mean_feat)/std_feat
  # reverse standardization
  pred = linear_model.predict(x_test)
  pred *= std_label
  pred += mean_label
  plt.plot(y_test,pred,'bo')

  plt.show()