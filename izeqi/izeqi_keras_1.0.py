import os,keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import advanced_activations
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras import backend as K
from sklearn import preprocessing
from keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
from keras.utils import plot_model
input_size_1=1
input_size_2=4

batchsize=100
#load the data
path = os.getcwd() + '\izeqi\data\data_train.csv'
data = pd.read_csv(path,header=None,names=['plotRatio',
                                             'transactionDate',
                                             'floorPrice','time',
                                             'price'])
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


path = os.getcwd() + '\izeqi\data\data_time.txt'
data2 = pd.read_csv(path,header='infer',index_col=0)
datatime=data2.values
x_train_time1= datatime[:, 0:input_size_1].astype(float)
y_train_time1 = datatime[:, input_size_1:]
scalertime_x = preprocessing.StandardScaler().fit(x_train_time1)
scalertime_y = preprocessing.StandardScaler().fit(y_train_time1)
x_train_time = scalertime_x.transform(x_train_time1)
y_train_time= scalertime_y.transform(y_train_time1)


model_time = LinearRegression()
model_time.fit(x_train_time, y_train_time)

'''
model_time=Sequential()
model_time.add(Dense(input_dim=1,kernel_initializer='random_uniform',
                bias_initializer='random_uniform',units=20))
model_time.add(keras.layers.advanced_activations.ELU(alpha=2))
model_time.add(Dropout(0.2))
model_time.add(Dense(1))

model_time.compile(optimizer='sgd',
              loss='mse')'''



#second model to calucate the price
model_value = Sequential()
#random init and zero_bias -----input layer of 4 units,and first layer is 17 units
model_value.add(Dense(17,kernel_initializer='random_uniform',
                bias_initializer='random_uniform',input_shape=(input_size_2,)))
model_value.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.3))

#second dropout layers
model_value.add(Dense(12))
model_value.add(Activation('selu'))
model_value.add(Dropout(0.1))
#third layers
model_value.add(Dense(15))
model_value.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.3))
model_value.add(Dropout(0.1))


#fifth layers
model_value.add(Dense(15))
model_value.add(Dropout(0.1))
#sixth layers
model_value.add(Dense(15))
#seventh layers
model_value.add(Dense(15))

model_value.add(Dropout(0.1))
model_value.add(Dense(100))
model_value.add(Activation('elu'))
#last layers_price

model_value.add(Dense(15))
model_value.add(Dropout(0.1))
model_value.add(Dense(15))
model_value.add(Dropout(0.1))
model_value.add(Dense(1))

# For a mean squared error regression problem
sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
model_value.compile(optimizer=sgd,
              loss='mae')




Saved=True
if(Saved==True):
    model_value = load_model('houseer_model.h5')
    model_value.load_weights('houseer_model_weights.h5')
    '''
    model_time = load_model('time_model.h5')
    model_time.load_weights('time_weights.h5')'''
    model_time = joblib.load('model_time.model')
# returns a compiled model
# identical to the previous one

else:
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='auto')
    model_value.fit(x_train, y_train,
                batch_size=batchsize, epochs=6, shuffle=True,
              callbacks=[early_stopping])
    joblib.dump(model_time, 'model_time.model')
    #model_time.fit(x_train_time,y_train_time, batch_size=batchsize)

    model_value.save('houseer_model.h5')  # creates a HDF5 file 'my_model.h5'
    model_value.save_weights('houseer_model_weights.h5')
    #model_time.save('time_model.h5')
    #model_time.save_weights('time_weights.h5')


y_pre=model_value.predict(x_test)
#restore data in test data
x_pre = scalerx.inverse_transform(x_test)
y_pre = scalery.inverse_transform(y_pre)



#load the new data
path = os.getcwd() + '\izeqi\data\data_new_1.txt'
data3 = pd.read_csv(path,header='infer',index_col=0)


#transform matrix type
datanew=data3.values
X_new_o = datanew[:, ].astype(float)
X_new_1=X_new_o[:,1].reshape([-1,1])
X_new_1= scalertime_x.transform(X_new_1)
get_time1=model_time.predict(X_new_1)
get_time = scalertime_y.inverse_transform(get_time1)
X_new_o=np.hstack((X_new_o,get_time))

def feedback(m):
    #这里显然不完整，要对月份有个加12的for
    y_pre=model_value.predict(m)
    #restore data in test data
    y_pre = scalery.inverse_transform(y_pre)
    return y_pre



a=np.zeros(shape=(120,len((X_new_o))))
X_new_a = X_new_o.copy()
for i in np.arange(0,12,0.1):

        X_new_a[:, 3]+=1 / 120
        X_new = scalerx.transform(X_new_a)
        y_new=feedback(X_new)
        a[int(i*10)]=y_new.reshape(1,-1)
for i in range(len(X_new)):
        # plot the time changge figure
        x1 = np.arange(0,12,0.1)
        x1=x1
        y1=a[:,i]
        y1=np.array(y1).astype(int)
        y1=y1.T
        y1=list(y1)
        fig=plt.figure(figsize=(7.5, 2.3))
        ax = fig.add_subplot(1, 1, 1)
        plt.grid()
        plt.xlim((0, 12))
        plt.xlabel('month')
        plt.ylabel('value')
        plt.xticks(np.linspace(0, 12, 13))
        plt.yticks(np.arange(min(y1), max(y1), (max(y1)-min(y1))/8))
        plt.plot(x1, y1, 'r')
        #plt.show()
        i=str(i)
        plt.savefig('D:\\pictures\\'+i+".png")

pd.DataFrame(X_new_o).to_csv("时间表.txt")
pd.DataFrame(a).to_csv("12个月价格表.txt")
