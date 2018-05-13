import os,keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras import backend as K
from sklearn import preprocessing
from keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


input_size_1=1
input_size_2=4

batchsize=70
#load the data
path = os.getcwd() + '\data\data_train.csv'
data = pd.read_csv(path, header=None, names=('plotRatio',
                                             'transactionDate',
                                             'floorPrice',
                                             'time','price'))
#transform matrix type
dataset=data.values
X = dataset[:, 0:input_size_2].astype(float)
Y = dataset[:, input_size_2:]
#normalize data
scalerx = preprocessing.StandardScaler().fit(X)
scalery= preprocessing.StandardScaler().fit(Y)
X = scalerx.transform(X)
Y = scalery.transform(Y)
#split data into trainlist and testlist
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, random_state=42)


path = os.getcwd() + '\data\time_train.csv'
data = pd.read_csv(path, header=None, names=('transactionDate',
                                             'time'))
datatime=data.values
x_train_time= datatime[:, 0:input_size_1].astype(float)
y_train_time = datatime[:, input_size_1:]
#need coding
#x_train_time=x_train[:,0]
#y_train_time=y_train[:,1]


model_time=Sequential()
model_time.add(Dense(5, activation='relu',kernel_initializer='random_uniform',
                bias_initializer='',input_shape=(input_size_1,)))
model_time.add(Dense(1))
sgd = SGD(lr=0.2, momentum=0.8, decay=0.0, nesterov=False)
model_time.compile(optimizer=sgd,
              loss='mean_squared_error')



#second model to calucate the price
model_value = Sequential()
#random init and zero_bias -----input layer of 4 units,and first layer is 17 units
model_value.add(Dense(17, activation='relu',kernel_initializer='random_uniform',
                bias_initializer='zeros',input_shape=(input_size_2,)))


#second dropout layers
model_value.add(Dense(12))
model_value.add(Activation('relu'))
#third layers
model_value.add(Dense(50))
model_value.add(Activation('relu'))
#fourth layers
model_value.add(Dense(64))
model_value.add(Activation('relu'))
#fifth layers
model_value.add(Dense(25))
model_value.add(Activation('relu'))
#sixth layers
model_value.add(Dense(50))
model_value.add(Activation('relu'))
#seventh layers
model_value.add(Dense(50))
model_value.add(Activation('relu'))
#eighth layers and dropout layers
model_value.add(Dropout(0.2))
model_value.add(Dense(100))
model_value.add(Activation('relu'))
#last layers_price
model_value.add(Dense(1))

# For a mean squared error regression problem
sgd = SGD(lr=0.1, momentum=0.8, decay=0.0, nesterov=False)
model_value.compile(optimizer=sgd,
              loss='mean_squared_error',metrics=['accuracy'])




Saved=False
if(Saved==True):
    model_value = load_model('houseer_model.h5')
    model_value.load_weights('houseer_model_weights.h5')

# returns a compiled model
# identical to the previous one

else:
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, mode='auto')
    model_value.fit(x_train, y_train,
                batch_size=batchsize, epochs=10, shuffle=True,
              callbacks=[early_stopping])
    model_time.fit(x_train_time,y_train_time)

    model_value.save('houseer_model.h5')  # creates a HDF5 file 'my_model.h5'
    model_value.save_weights('houseer_model_weights.h5')



y_pre=model_value.predict(x_test)
#restore data in test data
y_pre = scalery.inverse_transform(y_pre)


#load the new data
path = os.getcwd() + '\data\data_new_1.csv'
data = pd.read_csv(path, header=None, names=('plotRatio',
                                             'transactionDate',
                                             'floorPrice',
                                             ))


#transform matrix type
datanew=data.values
X_new = datanew[:, :].astype(float)
X_time=model_time.predict(X_new)
X_new=np.hstack(X_new,X_time)
X_new = scalerx.transform(X)

#这里显然不完整，要对月份有个加12的for
y_pre=model_value.predict(X_new)
#restore data in test data
y_pre = scalery.inverse_transform(y_pre)


#plot the time changge figure
x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
y1=[8000,9000,9500,9200,9300,8900,9300,9350,9300,9400,9600,10000,10048]
plot.figure(figsize=(7.5,2.3))
plot.grid()
plot.xlabel('month')
plot.ylabel('value')
plot.show()
