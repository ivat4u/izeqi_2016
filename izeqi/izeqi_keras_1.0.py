import os,keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras import backend as K
from sklearn import preprocessing
from keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



input_size_1=3
input_size_2=1
batchsize=70

path = os.getcwd() + '\data\data_1.csv'
data = pd.read_csv(path, header=None, names=None)
dataset=data.values

X = dataset[:, 0:input_size_1].astype(float)
X_train=X
Y = dataset[:, input_size_1:]
Y_train=Y
scalerx = preprocessing.StandardScaler().fit(X)
scalery= preprocessing.StandardScaler().fit(Y)
X = scalerx.transform(X)
Y = scalery.transform(Y)
model_value = Sequential()


model_value.add(Dense(17, activation='relu',kernel_initializer='random_uniform',
                bias_initializer='zeros',input_shape=(input_size_1,)))

model_value.add(Dropout(0.2))
model_value.add(Dense(50))
model_value.add(Activation('relu'))
model_value.add(Dense(50))
model_value.add(Activation('relu'))
model_value.add(Dense(50))
model_value.add(Activation('relu'))

model_value.add(Dense(1))

# For a mean squared error regression problem
sgd = SGD(lr=0.1, momentum=0.8, decay=0.0, nesterov=False)
model_value.compile(optimizer=sgd,
              loss='mean_squared_error',metrics=['accuracy'])





Saved=True
if(Saved==True):
    model = load_model('houseer_model.h5')
    model.load_weights('houseer_model_weights.h5')

# returns a compiled model
# identical to the previous one

else:
    x_train = X
    y_train = Y
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, mode='auto')
    model_value.fit(x_train, y_train,
                batch_size=batchsize, epochs=10, shuffle=True,
              callbacks=[early_stopping])

    model_value.save('houseer_model.h5')  # creates a HDF5 file 'my_model.h5'
    model_value.save_weights('houseer_model_weights.h5')

X_test=X
Y_test=Y
y_pre=model_value.predict(X_test)
y_pre = scalery.inverse_transform(y_pre)

x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
y1=[8000,9000,9500,9200,9300,8900,9300,9350,9300,9400,9600,10000,10048]



plot.figure(figsize=(7.5,2.3))
plot.grid()
plot.xlabel('month')
plot.ylabel('value')

plot.show()
