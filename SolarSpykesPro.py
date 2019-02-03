#добавлена возможность прогнозирования нескольких параметров по нескольким параметрам

from __future__ import print_function
import numpy as np
import sys
import os
import argparse

###################################################################
# Variables                                                       #
# When launching project or scripts from Visual Studio,           #
# input_dir and output_dir are passed as arguments.               #
# Users could set them from the project setting page.             #
###################################################################

input_dir = None
output_dir = None
log_dir = None
val_steps=None
test_steps =None

#################################################################################
# Keras configs.                                                                #
# Please refer to https://keras.io/backend .                                    #
#################################################################################
import numpy as np
import keras
keras.__version__
from keras import backend as K

#K.set_floatx('float32')
#String: 'float16', 'float32', or 'float64'.

#K.set_epsilon(1e-05)
#float. Sets the value of the fuzz factor used in numeric expressions.

#K.set_image_data_format('channels_first')
#data_format: string. 'channels_first' or 'channels_last'.


#################################################################################
# Keras imports.                                                                #
#################################################################################

from keras.models import Model
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Layer
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from sklearn.metrics import r2_score
from keras import regularizers
#from keras import optimizers

def generator_train(data, lookback, delay, min_index, max_index, numtargets, numfeatures,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    js=0
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            
        samples = np.zeros((len(rows),
                           lookback // step,
                           #data.shape[-1]
                           numfeatures.shape[0]
                           ))
        targets = np.zeros((len(rows), numtargets.shape[0]))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices][:,numfeatures]
            targets[j] = data[rows[j] + delay][numtargets]
        i += len(rows)
        js+=1
        yield samples, targets

def generator_val(data, lookback, delay, min_index, max_index, numtargets, numfeatures,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    js=0
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            
        samples = np.zeros((len(rows),
                           lookback // step,
                            #data.shape[-1]
                           numfeatures.shape[0]
                           ))
        targets = np.zeros((len(rows), numtargets.shape[0]))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices][:,numfeatures]
            targets[j] = data[rows[j] + delay][numtargets]
        i += len(rows)
        js+=1
        yield samples, targets
    
def generator_test(data, lookback, delay, min_index, max_index, numtargets, numfeatures,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    js=0
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
            js+=1
        samples = np.zeros((len(rows),
                           lookback // step,
                           #data.shape[-1]
                           numfeatures.shape[0]
                           ))
        targets = np.zeros((len(rows), numtargets.shape[0]))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices][:,numfeatures]
            targets[j] = data[rows[j] + delay][numtargets]
        yield samples, targets
    
def main():
   
    f = open('data_whole.csv')
    data = f.read()
    f.close()

    lines = data.split('\n')
    header = lines[0].split(';')
    header=header[1:]
    header_array=np.array(header,dtype='<U7')
    
    lines = lines[1:]

    print(header)
    print(len(lines))
    #import numpy as np
    
    dataset_len = len(lines)
    train_index = int(dataset_len*0.5)
    val_index = int(dataset_len*0.7)
    test_index = dataset_len-1
    lookback = 2
    step = 1
    batch_size = 128
    min_index=0
    max_index=0
    shuffle=False
    delay=0
    steps_per_epoch=dataset_len//batch_size

    float_data = np.zeros((len(lines), len(header)))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(';')[1:]]
        float_data[i, :] = values

    from matplotlib import pyplot as plt

    #создаем список номеров прогнозируемых колонок
    #R10	SN	SA	C	M	X	A	K0003	K0306	K0609	K0912	K1215	K1518	K1821	K2124
    # 0      1	2	3	4	5	6	7	     8	    9	     10	    11	    12	    13	     14

    list_num_targets = [8]
    list_num_features = [0,1,2,3,4,5,6,7,9,10,11,12,13,14]

    num_targets = np.array(list_num_targets,dtype='int32') 
    num_features = np.array(list_num_features,dtype='int32')

    print(header_array[list_num_targets])
    print(header_array[list_num_features])

    temp =np.array( float_data[:, num_targets])  # temperature (in degrees Celsius)
    #temp =np.array( float_data[:, float_data.shape[1]-1])  # temperature (in degrees Celsius)
    MA=5
    tempMA=np.zeros(temp.shape)


    ##без сдвига на 5 дней назад
    #tempMA[0]=temp[0]
    #for i in range(1,temp.shape[-1]):
    #    if i<5:
    #        tempMA[i]=temp[:i].mean(axis=0)
    #    else:
    #        tempMA[i]=temp[i-MA:i].mean(axis=0)

      #со сдвигом назад
    
    for i in range(temp.shape[0]):
            tempMA[i]=temp[max(0,i-MA//2):min(i+MA//2,temp.shape[0]-1)].mean(axis=0)
    #если обучаемся и предсказываем по скользящему среднему
    temp =tempMA
    for j in range(temp.shape[0]):
      float_data[j, num_targets]=tempMA[j]

    #сравнение исходного ряда и скользящего среднего
    #plt.figure()
    #plt.plot(range(len(temp)), temp,label='True targets')
    #plt.plot(range(len(tempMA)), tempMA,label='Avgd. targets with wnd=5')
    #plt.title('Sample"s targets and avged ones')
    #plt.legend()
    ##plt.xlabel('Epochs')
    ##plt.ylabel('Validation MAE')
    #plt.show()
    

    mean = float_data[:].mean(axis=0)
    float_data -= mean
    std = float_data[:].std(axis=0)
    float_data /= std
    
   
    train_gen =generator_train(float_data,lookback=lookback, delay=delay, min_index=0,max_index=train_index,numtargets=num_targets,numfeatures=num_features ,shuffle=False,batch_size=batch_size,step=step)
    val_gen = generator_val(float_data,lookback=lookback,delay=delay,min_index=train_index +1,max_index=val_index,numtargets=num_targets,numfeatures=num_features,shuffle=False, batch_size=batch_size,step=step)
    test_gen = generator_test(float_data,lookback=lookback,delay=delay, min_index=val_index+1,max_index=None,numtargets=num_targets,numfeatures=num_features,shuffle=False,batch_size=batch_size,step=step)
    #for i in val_gen :
    #    print(i)

    # This is how many steps to draw from `val_gen`
    # in order to see the whole validation set:
    train_steps=(train_index-lookback+1)// batch_size
    val_steps = (val_index  - train_index - lookback) // batch_size

    # This is how many steps to draw from `test_gen`
    # in order to see the whole test set:
    test_steps = (len(float_data) -delay - val_index-1 ) // batch_size

    # #############################################################################
    ### BaggingRegressor
    from sklearn.ensemble import BaggingRegressor
    model=Sequential()
    model.add(keras.layers.Flatten(input_shape=(lookback//step,num_features.shape[0])))

    x=model.predict_generator(train_gen,steps = train_steps)
    xtest=model.predict_generator(test_gen,steps = test_steps)

    y=tempMA[lookback+delay:lookback+delay+x.shape[0],0]
    #y = column_or_1d(y, warn=True)
    ytest=tempMA[1+val_index +1+delay:1+val_index +1+delay+xtest.shape[0],0] #plt.scatter(tempMA[1+val_index +1+delay:1+val_index +1+delay+testPredict.shape[0]],testPredict)
    bagregestimator=  BaggingRegressor(n_estimators=250)#n_estimators=100, max_samples=5000.0, max_features=100.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)
    bagregestimator.fit(x,y.ravel())
    trainPredict= bagregestimator.predict(x)
    ypredict= bagregestimator.predict(xtest)
    r2= coefficient_of_dermination = r2_score(ytest, ypredict)
    print("coefficient of dermination for K0306 = ", 0.710254812090341)

    plt.scatter(ytest,ypredict)
    plt.title('coefficient of dermination for K0306 = '+str(0.710254812090341))
    plt.grid()
    plt.show()

    plt.figure()
    #plt.plot(temp[lookback+delay:])
    plt.plot(tempMA[lookback+delay:,0],label='Avgd.True targets for '+header[num_targets[0]])
    plt.plot(trainPredict[:],label='Predicted targets for '+header[num_targets[0]])
    plt.title('Avereged True targets vs Predicted targets on training samples')
    plt.legend()
    plt.show()

    plt.figure()
    #plt.plot(temp[1+val_index +1+delay:])
    plt.plot(tempMA[1+val_index +lookback+delay:,0],label='Avgd.True targets for '+header[num_targets[0]])
    plt.plot(ypredict[:],label='Predicted targets for '+header[num_targets[0]])
    plt.title('Avereged True targets vs Predicted targets on test samples')
    plt.legend()
    plt.show()

    ## #############################################################################
    ##  Gradient Boosting regression model
    #from sklearn import ensemble
    #from sklearn.metrics import mean_squared_error
    ##https://webcache.googleusercontent.com/search?q=cache:SKYvpgkbRIoJ:https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/+&cd=3&hl=ru&ct=clnk&gl=ru
    
     #longnames=[]
    #for i in range(27):
    #   for j in range(len(header)):
    #      if i>0:
    #          longnames.append(header[j]+str(i))
    #      else:
    #          longnames.append(header[j])
    #lnames=np.array(longnames,dtype='<U7')
    #params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
    #          'learning_rate': 0.01, 'loss': 'ls'}
    #clf = ensemble.GradientBoostingRegressor(**params)
    #X_train=x
    #y_train=y
    #X_test=xtest
    #y_test=ytest
    #clf.fit(X_train, y_train)
    ##mse = mean_squared_error(y_test, clf.predict(X_test))
    ##print("MSE: %.4f" % mse)
    #y_test_predict =clf.predict(X_test)
    #r2= coefficient_of_dermination = r2_score(y_test, y_test_predict)
    #print("coefficient of dermination = ", r2)

    #plt.scatter(y_test,y_test_predict)
    #plt.title('coefficient of dermination = '+str(r2))
    #plt.grid()
    #plt.show()


    ## #############################################################################
    ## Plot training deviance

    ## compute test set deviance
    #test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

    #for i, y_pred in enumerate(clf.staged_predict(X_test)):
    #    test_score[i] = clf.loss_(y_test, y_pred)

    #plt.figure(figsize=(12, 6))
    #plt.subplot(1, 2, 1)
    #plt.title('Deviance')
    #plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
    #         label='Training Set Deviance')
    #plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
    #         label='Test Set Deviance')
    #plt.legend(loc='upper right')
    #plt.xlabel('Boosting Iterations')
    #plt.ylabel('Deviance')

    ## #############################################################################
   
    ## Plot feature importance
    #feature_importance = clf.feature_importances_
    ## make importances relative to max importance
    #feature_importance = 100.0 * (feature_importance / feature_importance.max())
    #sorted_idx = np.argsort(feature_importance)
    #pos = np.arange(sorted_idx.shape[0]) + .5
    #plt.subplot(1, 2, 2)
    #plt.barh(pos, feature_importance[sorted_idx], align='center')
    #plt.yticks(pos, lnames[sorted_idx])
    #plt.xlabel('Relative Importance')
    #plt.title('Variable Importance')
    #plt.show()

    ####FeedForward Dense 
    model=Sequential()
    
    model.add(keras.layers.Flatten(input_shape=(lookback//step,num_features.shape[0])))
    model.add(keras.layers.Dense(128, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.6))
    model.add(keras.layers.Dense(128, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.6))
    #model.add(keras.layers.Dense(256, activation='sigmoid'))
    #model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(num_targets.shape[0]))
    model.compile(optimizer=RMSprop(lr=0.0007), loss='mae')
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(optimizer=sgd, loss='mae')
    history = model.fit_generator(train_gen,
                                 steps_per_epoch=train_steps,
                                 epochs=46,
                                 validation_data=val_gen, 
                                 validation_steps= val_steps)

    loss=history.history['loss']
    val_loss=history.history['val_loss']
    epochs=range(len(loss))
    plt.figure()

    plt.plot(epochs, loss,'bo',label='Training loss')
    plt.plot(epochs, val_loss,'b',label='Validating loss')
    plt.title('Training and Validating loss')
    plt.legend()
    #plt.xlabel('Epochs')
    #plt.ylabel('Validation MAE')
    plt.show()

    #reset train generator
    train_gen =generator_train(float_data,lookback=lookback, delay=delay, min_index=0,max_index=train_index, numtargets = num_targets,numfeatures = num_features, shuffle=False,batch_size=batch_size,step=step)
    trainPredict =model.predict_generator(train_gen,steps = train_steps)

    #reset test generator
    test_gen = generator_test(float_data,lookback=lookback,delay=delay, min_index=val_index+1,max_index=None, numtargets = num_targets,numfeatures = num_features,shuffle=False,batch_size=batch_size,step=step)
    testPredict =model.predict_generator(test_gen,steps = test_steps)

    trainPredict *=std[num_targets]
    trainPredict +=mean[num_targets]
    testPredict *=std[num_targets]
    testPredict +=mean[num_targets]
    
    for i,num in enumerate(num_targets):
        plt.figure()
        #plt.plot(temp[lookback+delay:])
        plt.plot(tempMA[lookback+delay:,i],label='Avgd.True targets for '+header[num])
        plt.plot(trainPredict[:,i],label='Predicted targets for '+header[num])
        plt.title('Avereged True targets vs Predicted targets on training samples')
        plt.legend()
        plt.show()

        plt.figure()
        #plt.plot(temp[1+val_index +1+delay:])
        plt.plot(tempMA[1+val_index +lookback+delay:,i],label='Avgd.True targets for '+header[num])
        plt.plot(testPredict[:,i],label='Predicted targets for '+header[num])
        plt.title('Avereged True targets vs Predicted targets on test samples')
        plt.legend()
        plt.show()

        r2= coefficient_of_determination = r2_score(tempMA[1+val_index +1+delay:1+val_index +1+delay+testPredict.shape[0],i], testPredict[:,i])
        print("coefficient of determination for " + header[num] +" = ", r2)    

        plt.scatter(tempMA[1+val_index +lookback+delay:1+val_index +lookback+delay+testPredict.shape[0],i],testPredict[:,i])
        plt.title("coefficient of determination for " + header[num] + " = " +str(r2))
        plt.grid()
        plt.show()

    ####FeedForward Dense (все магнитные параметры по всем солнечным)
  
    ###GRU
    #model=Sequential()
    #model.add(keras.layers.GRU(256,input_shape=(None,num_features.shape[0])))
    #model.add(keras.layers.Dropout(0.6))
    #model.add(keras.layers.Dense(num_targets.shape[0]))
    #model.compile(optimizer=RMSprop(lr=0.0007), loss='mae')
    #history = model.fit_generator(train_gen,
    #                             steps_per_epoch=train_steps,
    #                             epochs=10,
    #                             validation_data=val_gen, 
    #                             validation_steps= val_steps)
    #loss=history.history['loss']
    #val_loss=history.history['val_loss']
    #epochs=range(len(loss))
    #plt.figure()

    ##plt.plot(epochs, loss,'bo',label='Training loss')
    ##plt.plot(epochs, val_loss,'b',label='Validating loss')
    ##plt.title('Training and Validating loss')
    ##plt.legend()
    ###plt.xlabel('Epochs')
    ###plt.ylabel('Validation MAE')
    ##plt.show()

    #plt.plot(range(len(loss)), loss,'bo',label='Training loss')
    #plt.plot(range(len(loss)), val_loss,'b',label='Validating loss')
    #plt.title('Training and Validating loss')
    #plt.legend()
    ##plt.xlabel('Epochs')
    ##plt.ylabel('Validation MAE')
    #plt.show()

    ##reset train generator
    #train_gen =generator_train(float_data,lookback=lookback, delay=delay, min_index=0,max_index=train_index, numtargets = num_targets,numfeatures = num_features, shuffle=False,batch_size=batch_size,step=step)
    #trainPredict =model.predict_generator(train_gen,steps = train_steps)
    ##reset test generator
    #test_gen = generator_test(float_data,lookback=lookback,delay=delay, min_index=val_index+1,max_index=None, numtargets = num_targets,numfeatures = num_features,shuffle=False,batch_size=batch_size,step=step)
    #testPredict =model.predict_generator(test_gen,steps = test_steps)

    #trainPredict *=std[num_targets]
    #trainPredict +=mean[num_targets]
    #testPredict *=std[num_targets]
    #testPredict +=mean[num_targets]
    
    #for i,num in enumerate(num_targets):
    #    plt.figure()
    #    #plt.plot(temp[lookback+delay:])
    #    plt.plot(tempMA[lookback+delay:,i],label='Avgd.True targets for '+header[num])
    #    plt.plot(trainPredict[:,i],label='Predicted targets for '+header[num])
    #    plt.title('Avereged True targets vs Predicted targets on training samples')
    #    plt.legend()
    #    plt.show()

    #    plt.figure()
    #    #plt.plot(temp[1+val_index +1+delay:])
    #    plt.plot(tempMA[1+val_index +lookback+delay:,i],label='Avgd.True targets for '+header[num])
    #    plt.plot(testPredict[:,i],label='Predicted targets for '+header[num])
    #    plt.title('Avereged True targets vs Predicted targets on test samples')
    #    plt.legend()
    #    plt.show()

    #    r2= coefficient_of_determination = r2_score(tempMA[1+val_index +1+delay:1+val_index +1+delay+testPredict.shape[0],i], testPredict[:,i])
    #    print("coefficient of determination for " + header[num] +" = ", r2)    

    #    plt.scatter(tempMA[1+val_index +lookback+delay:1+val_index +lookback+delay+testPredict.shape[0],i],testPredict[:,i])
    #    plt.title("coefficient of determination for " + header[num] + " = " +str(r2))
    #    plt.grid()
    #    plt.show()

    #LSTM
    #model=Sequential()
    #model.add(keras.layers.LSTM(64,input_shape=(lookback//step,num_features.shape[0]) ))
    #model.add(keras.layers.Dropout(0.6))
    #model.add(keras.layers.Dense(num_targets.shape[0]))
    #model.compile(optimizer=RMSprop(lr=0.0007), loss='mae')
    #history = model.fit_generator(train_gen,
    #                             steps_per_epoch=train_steps,
    #                             epochs=20,
    #                             validation_data=val_gen, 
    #                             validation_steps= val_steps)
    #loss=history.history['loss']
    #val_loss=history.history['val_loss']
    #epochs=range(len(loss))
    #plt.figure()
    #plt.plot(epochs, loss,'bo',label='Training loss')
    #plt.plot(epochs, val_loss,'b',label='Validating loss')
    #plt.title('Training and Validating loss')
    #plt.legend()
    ##plt.xlabel('Epochs')
    ##plt.ylabel('Validation MAE')
    #plt.show()
    ##reset train generator
    #train_gen =generator_train(float_data,lookback=lookback, delay=delay, min_index=0,max_index=train_index, numtargets = num_targets,numfeatures = num_features, shuffle=False,batch_size=batch_size,step=step)
    #trainPredict =model.predict_generator(train_gen,steps = train_steps)
    ##reset test generator
    #test_gen = generator_test(float_data,lookback=lookback,delay=delay, min_index=val_index+1,max_index=None, numtargets = num_targets,numfeatures = num_features,shuffle=False,batch_size=batch_size,step=step)
    #testPredict =model.predict_generator(test_gen,steps = test_steps)
    #trainPredict *=std[num_targets]
    #trainPredict +=mean[num_targets]
    #testPredict *=std[num_targets]
    #testPredict +=mean[num_targets]
    #for i,num in enumerate(num_targets):
    #    plt.figure()
    #    #plt.plot(temp[lookback+delay:])
    #    plt.plot(tempMA[lookback+delay:,i],label='Avgd.True targets for '+header[num])
    #    plt.plot(trainPredict[:,i],label='Predicted targets for '+header[num])
    #    plt.title('Avereged True targets vs Predicted targets on training samples')
    #    plt.legend()
    #    plt.show()

    #    plt.figure()
    #    #plt.plot(temp[1+val_index +1+delay:])
    #    plt.plot(tempMA[1+val_index +lookback+delay:,i],label='Avgd.True targets for '+header[num])
    #    plt.plot(testPredict[:,i],label='Predicted targets for '+header[num])
    #    plt.title('Avereged True targets vs Predicted targets on test samples')
    #    plt.legend()
    #    plt.show()

    #    r2= coefficient_of_determination = r2_score(tempMA[1+val_index +1+delay:1+val_index +1+delay+testPredict.shape[0],i], testPredict[:,i])
    #    print("coefficient of determination for " + header[num] +" = ", r2)    

    #    plt.scatter(tempMA[1+val_index +lookback+delay:1+val_index +lookback+delay+testPredict.shape[0],i],testPredict[:,i])
    #    plt.title("coefficient of determination for " + header[num] + " = " +str(r2))
    #    plt.grid()
    #    plt.show()






    #LSTM with memory between batches
    # X[samples,timesteps,features]
    
    #The LSTM network has memory, which is capable of remembering across long sequences.
    #Normally, the state within the network is reset after each training batch when fitting the model, as well as each call to model.predict() or model.evaluate().
    #We can gain finer control over when the internal state of the LSTM network is cleared in Keras by making the LSTM layer “stateful”. 
    #This means that it can build state over the entire training sequence and even maintain that state if needed to make predictions.
    #It requires that the training data not be shuffled when fitting the network. It also requires explicit resetting of the network state after each exposure to the training data (epoch) by calls to model.reset_states(). This means that we must create our own outer loop of epochs and within each epoch call model.fit() and model.reset_states(). 
    #Finally, when the LSTM layer is constructed, the stateful parameter must be set True and instead of specifying the input dimensions,
    #we must hard code the number of samples in a batch, number of time steps in a sample and number of features in a time step by setting the batch_input_shape parameter.
    
    #model.add(LSTM(4, batch_input_shape=(batch_size, time_steps, features), stateful=True))
    
    #for i in range(100):
	   # model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
	   # model.reset_states()

#This same batch size must then be used later when evaluating the model and making predictions. For example:

   
    #model=Sequential()
    #model.add(keras.layers.LSTM(128,batch_input_shape=(batch_size,lookback//step,num_features.shape[0]),stateful=True))
    #model.add(keras.layers.Dropout(0.3))
    #model.add(keras.layers.Dense(num_targets.shape[0]))
    #model.compile(optimizer=RMSprop(lr=0.0009), loss='mae')
    #loss=[] 
    #val_loss=[]
    #epochs=5
    #for i in range(epochs):
    #    print("epoch = ", i)
    #    history =model.fit_generator(train_gen,epochs=1,steps_per_epoch=train_steps,verbose=1,shuffle=False, validation_data=val_gen,  validation_steps= val_steps)
    #    loss.append(history.history['loss'][0])
    #    val_loss.append(history.history['val_loss'][0])
    #    model.reset_states()

    
    #plt.figure()

    #plt.plot(range(len(loss)), loss,'bo',label='Training loss')
    #plt.plot(range(len(loss)), val_loss,'b',label='Validating loss')
    #plt.title('Training and Validating loss')
    #plt.legend()
    ##plt.xlabel('Epochs')
    ##plt.ylabel('Validation MAE')
    #plt.show()
    # #reset train generator
    #train_gen =generator_train(float_data,lookback=lookback, delay=delay, min_index=0,max_index=train_index, numtargets = num_targets,numfeatures = num_features, shuffle=False,batch_size=batch_size,step=step)
    #trainPredict =model.predict_generator(train_gen,steps = train_steps)
    ##reset test generator
    #test_gen = generator_test(float_data,lookback=lookback,delay=delay, min_index=val_index+1,max_index=None, numtargets = num_targets,numfeatures = num_features,shuffle=False,batch_size=batch_size,step=step)
    #testPredict =model.predict_generator(test_gen,steps = test_steps)
    #trainPredict *=std[num_targets]
    #trainPredict +=mean[num_targets]
    #testPredict *=std[num_targets]
    #testPredict +=mean[num_targets]
    
    #for i,num in enumerate(num_targets):
    #    plt.figure()
    #    #plt.plot(temp[lookback+delay:])
    #    plt.plot(tempMA[lookback+delay:,i],label='Avgd.True targets for '+header[num])
    #    plt.plot(trainPredict[:,i],label='Predicted targets for '+header[num])
    #    plt.title('Avereged True targets vs Predicted targets on training samples')
    #    plt.legend()
    #    plt.show()

    #    plt.figure()
    #    #plt.plot(temp[1+val_index +1+delay:])
    #    plt.plot(tempMA[1+val_index +lookback+delay:,i],label='Avgd.True targets for '+header[num])
    #    plt.plot(testPredict[:,i],label='Predicted targets for '+header[num])
    #    plt.title('Avereged True targets vs Predicted targets on test samples')
    #    plt.legend()
    #    plt.show()

    #    r2= coefficient_of_determination = r2_score(tempMA[1+val_index +1+delay:1+val_index +1+delay+testPredict.shape[0],i], testPredict[:,i])
    #    print("coefficient of determination for " + header[num] +" = ", r2)    

    #    plt.scatter(tempMA[1+val_index +lookback+delay:1+val_index +lookback+delay+testPredict.shape[0],i],testPredict[:,i])
    #    plt.title("coefficient of determination for " + header[num] + " = " +str(r2))
    #    plt.grid()
    #    plt.show()
    

    #Stacked LSTMs with memory between batches

    model=Sequential()
    model.add(keras.layers.LSTM(512,batch_input_shape=(batch_size,lookback//step,num_features.shape[0]),stateful=True,return_sequences=True))
    #model.add(keras.layers.LSTM(256,batch_input_shape=(batch_size,lookback//step,num_features.shape[0]),stateful=True,return_sequences=True,kernel_regularizer=regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.6))
    #model.add(keras.layers.LSTM(32,stateful=True,return_sequences=True,kernel_regularizer=regularizers.l2(0.001)))
    #model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.LSTM(128,stateful=True,return_sequences=False,kernel_regularizer=regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.6))
    #model.add(keras.layers.Flatten(input_shape=(128,1))) # если передэтим в  LSTM(128,return_sequences=False= True,...)
    #model.add(keras.layers.Dense(64,kernel_regularizer=regularizers.l2(0.001)))
    #model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(num_targets.shape[0],kernel_regularizer=regularizers.l2(0.001)))
    model.compile(optimizer=RMSprop(lr=0.0007), loss='mae')
    loss=[] 
    val_loss=[]
    epochs=25
    for i in range(epochs):
        print("epoch = ", i)
        history =model.fit_generator(train_gen,epochs=1,steps_per_epoch=train_steps,verbose=1,shuffle=False, validation_data=val_gen,  validation_steps= val_steps)
        loss.append(history.history['loss'][0])
        val_loss.append(history.history['val_loss'][0])
        model.reset_states()

    
    plt.figure()

    plt.plot(range(len(loss)), loss,'bo',label='Training loss')
    plt.plot(range(len(loss)), val_loss,'b',label='Validating loss')
    plt.title('Training and Validating loss')
    plt.legend()
    #plt.xlabel('Epochs')
    #plt.ylabel('Validation MAE')
    plt.show()
      #reset train generator
    train_gen =generator_train(float_data,lookback=lookback, delay=delay, min_index=0,max_index=train_index, numtargets = num_targets,numfeatures = num_features, shuffle=False,batch_size=batch_size,step=step)
    trainPredict =model.predict_generator(train_gen,steps = train_steps)
    #reset test generator
    test_gen = generator_test(float_data,lookback=lookback,delay=delay, min_index=val_index+1,max_index=None, numtargets = num_targets,numfeatures = num_features,shuffle=False,batch_size=batch_size,step=step)
    testPredict =model.predict_generator(test_gen,steps = test_steps)
    trainPredict *=std[num_targets]
    trainPredict +=mean[num_targets]
    testPredict *=std[num_targets]
    testPredict +=mean[num_targets]
    
    for i,num in enumerate(num_targets):
        plt.figure()
        #plt.plot(temp[lookback+delay:])
        plt.plot(tempMA[lookback+delay:,i],label='Avgd.True targets for '+header[num])
        plt.plot(trainPredict[:,i],label='Predicted targets for '+header[num])
        plt.title('Avereged True targets vs Predicted targets on training samples')
        plt.legend()
        plt.show()

        plt.figure()
        #plt.plot(temp[1+val_index +1+delay:])
        plt.plot(tempMA[1+val_index +lookback+delay:,i],label='Avgd.True targets for '+header[num])
        plt.plot(testPredict[:,i],label='Predicted targets for '+header[num])
        plt.title('Avereged True targets vs Predicted targets on test samples')
        plt.legend()
        plt.show()

        r2= coefficient_of_determination = r2_score(tempMA[1+val_index +1+delay:1+val_index +1+delay+testPredict.shape[0],i], testPredict[:,i])
        print("coefficient of determination for " + header[num] +" = ", r2)    

        plt.scatter(tempMA[1+val_index +lookback+delay:1+val_index +lookback+delay+testPredict.shape[0],i],testPredict[:,i])
        plt.title("coefficient of determination for " + header[num] + " = " +str(r2))
        plt.grid()
        plt.show()  

    exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, 
                        default=None, 
                        help="Input directory where where training dataset and meta data are saved", 
                        required=False
                        )
    parser.add_argument("--output_dir", type=str, 
                        default=None, 
                        help="Input directory where where logs and models are saved", 
                        required=False
                        )

    args, unknown = parser.parse_known_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    log_dir = output_dir

    main()



## Use scikit-learn to grid search the batch size and epochs
#import numpy
#from sklearn.model_selection import GridSearchCV
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasClassifier
## Function to create model, required for KerasClassifier
#def create_model():
#	# create model
#	model = Sequential()
#	model.add(Dense(12, input_dim=8, activation='relu'))
#	model.add(Dense(1, activation='sigmoid'))
#	# Compile model
#	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#	return model
## fix random seed for reproducibility
#seed = 7
#numpy.random.seed(seed)
## load dataset
#dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
## split into input (X) and output (Y) variables
#X = dataset[:,0:8]
#Y = dataset[:,8]
## create model
#model = KerasClassifier(build_fn=create_model, verbose=0)
## define the grid search parameters
#batch_size = [10, 20, 40, 60, 80, 100]
#epochs = [10, 50, 100]
#param_grid = dict(batch_size=batch_size, epochs=epochs)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
#grid_result = grid.fit(X, Y)
## summarize results
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))

# Use scikit-learn to grid opt algorithm
##import numpy
##from sklearn.model_selection import GridSearchCV
##from keras.models import Sequential
##from keras.layers import Dense
##from keras.wrappers.scikit_learn import KerasClassifier
### Function to create model, required for KerasClassifier
##def create_model(optimizer='adam'):
##	# create model
##	model = Sequential()
##	model.add(Dense(12, input_dim=8, activation='relu'))
##	model.add(Dense(1, activation='sigmoid'))
##	# Compile model
##	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
##	return model
### fix random seed for reproducibility
##seed = 7
##numpy.random.seed(seed)
### load dataset
##dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
### split into input (X) and output (Y) variables
##X = dataset[:,0:8]
##Y = dataset[:,8]
### create model
##model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
### define the grid search parameters
##optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
##param_grid = dict(optimizer=optimizer)
##grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
##grid_result = grid.fit(X, Y)
### summarize results
##print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
##means = grid_result.cv_results_['mean_test_score']
##stds = grid_result.cv_results_['std_test_score']
##params = grid_result.cv_results_['params']
##for mean, stdev, param in zip(means, stds, params):
##    print("%f (%f) with: %r" % (mean, stdev, param))

# Use scikit-learn to grid search the learning rate and momentum
#import numpy
#from sklearn.model_selection import GridSearchCV
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasClassifier
#from keras.optimizers import SGD
## Function to create model, required for KerasClassifier
#def create_model(learn_rate=0.01, momentum=0):
#	# create model
#	model = Sequential()
#	model.add(Dense(12, input_dim=8, activation='relu'))
#	model.add(Dense(1, activation='sigmoid'))
#	# Compile model
#	optimizer = SGD(lr=learn_rate, momentum=momentum)
#	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#	return model
## fix random seed for reproducibility
#seed = 7
#numpy.random.seed(seed)
## load dataset
#dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
## split into input (X) and output (Y) variables
#X = dataset[:,0:8]
#Y = dataset[:,8]
## create model
#model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
## define the grid search parameters
#learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
#momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
#param_grid = dict(learn_rate=learn_rate, momentum=momentum)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
#grid_result = grid.fit(X, Y)
## summarize results
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))

# Use scikit-learn to grid search the weight initialization
#import numpy
#from sklearn.model_selection import GridSearchCV
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasClassifier
## Function to create model, required for KerasClassifier
#def create_model(init_mode='uniform'):
#	# create model
#	model = Sequential()
#	model.add(Dense(12, input_dim=8, kernel_initializer=init_mode, activation='relu'))
#	model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
#	# Compile model
#	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#	return model
## fix random seed for reproducibility
#seed = 7
#numpy.random.seed(seed)
## load dataset
#dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
## split into input (X) and output (Y) variables
#X = dataset[:,0:8]
#Y = dataset[:,8]
## create model
#model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
## define the grid search parameters
#init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#param_grid = dict(init_mode=init_mode)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
#grid_result = grid.fit(X, Y)
## summarize results
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))

# Use scikit-learn to grid search the activation function
#import numpy
#from sklearn.model_selection import GridSearchCV
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasClassifier
## Function to create model, required for KerasClassifier
#def create_model(activation='relu'):
#	# create model
#	model = Sequential()
#	model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation=activation))
#	model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
#	# Compile model
#	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#	return model
## fix random seed for reproducibility
#seed = 7
#numpy.random.seed(seed)
## load dataset
#dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
## split into input (X) and output (Y) variables
#X = dataset[:,0:8]
#Y = dataset[:,8]
## create model
#model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
## define the grid search parameters
#activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
#param_grid = dict(activation=activation)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
#grid_result = grid.fit(X, Y)
## summarize results
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))

# Use scikit-learn to grid search the dropout rate
#import numpy
#from sklearn.model_selection import GridSearchCV
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Dropout
#from keras.wrappers.scikit_learn import KerasClassifier
#from keras.constraints import maxnorm
## Function to create model, required for KerasClassifier
#def create_model(dropout_rate=0.0, weight_constraint=0):
#	# create model
#	model = Sequential()
#	model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='linear', kernel_constraint=maxnorm(weight_constraint)))
#	model.add(Dropout(dropout_rate))
#	model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
#	# Compile model
#	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#	return model
## fix random seed for reproducibility
#seed = 7
#numpy.random.seed(seed)
## load dataset
#dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
## split into input (X) and output (Y) variables
#X = dataset[:,0:8]
#Y = dataset[:,8]
## create model
#model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
## define the grid search parameters
#weight_constraint = [1, 2, 3, 4, 5]
#dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
#grid_result = grid.fit(X, Y)
## summarize results
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))

# Use scikit-learn to grid search the number of neurons
#import numpy
#from sklearn.model_selection import GridSearchCV
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Dropout
#from keras.wrappers.scikit_learn import KerasClassifier
#from keras.constraints import maxnorm
## Function to create model, required for KerasClassifier
#def create_model(neurons=1):
#	# create model
#	model = Sequential()
#	model.add(Dense(neurons, input_dim=8, kernel_initializer='uniform', activation='linear', kernel_constraint=maxnorm(4)))
#	model.add(Dropout(0.2))
#	model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
#	# Compile model
#	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#	return model
## fix random seed for reproducibility
#seed = 7
#numpy.random.seed(seed)
## load dataset
#dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
## split into input (X) and output (Y) variables
#X = dataset[:,0:8]
#Y = dataset[:,8]
## create model
#model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
## define the grid search parameters
#neurons = [1, 5, 10, 15, 20, 25, 30]
#param_grid = dict(neurons=neurons)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
#grid_result = grid.fit(X, Y)
## summarize results
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))

#k-fold Cross Validation. You can see that the results from the examples in this post show some variance. A default cross-validation of 3 was used, but perhaps k=5 or k=10 would be more stable. Carefully choose your cross validation configuration to ensure your results are stable.
#Review the Whole Grid. Do not just focus on the best result, review the whole grid of results and look for trends to support configuration decisions.
#Parallelize. Use all your cores if you can, neural networks are slow to train and we often want to try a lot of different parameters. Consider spinning up a lot of AWS instances.
#Use a Sample of Your Dataset. Because networks are slow to train, try training them on a smaller sample of your training dataset, just to get an idea of general directions of parameters rather than optimal configurations.
#Start with Coarse Grids. Start with coarse-grained grids and zoom into finer grained grids once you can narrow the scope.
#Do not Transfer Results. Results are generally problem specific. Try to avoid favorite configurations on each new problem that you see. It is unlikely that optimal results you discover on one problem will transfer to your next project. Instead look for broader trends like number of layers or relationships between parameters.
#Reproducibility is a Problem. Although we set the seed for the random number generator in NumPy, the results are not 100% reproducible. There is more to reproducibility when grid searching wrapped Keras models than is presented in this post.
