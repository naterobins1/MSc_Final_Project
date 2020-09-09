"""

lets build some models!!!

"""
#imports
from keras import models
from keras import layers
from keras.optimizers import RMSprop, Adagrad
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, GRU, LayerNormalization
from keras.utils.np_utils import to_categorical
from keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import time
import matplotlib

##################################################################################################

#DATA PROCESSING

def load_all_data(include_time=bool,catergor=True,normalise = False):
    """

    function written on 13th of July to load all of the data I obtained last week (OU, Colby, Anne-Marie & Brenna)
    :return: CVs is an array of all the CVs, and also the respective labels (i.e. the concentration)
    """

    rootdir = "/Users/nathanrobins/Documents/MSc_Neurotech/MSc_Project/Data/All_PostCal"
    arr = os.listdir(rootdir)
    count = 0

    #initially go through all the data and find out how many CV files there are
    for folder in arr:
        if folder != '.DS_Store':
            newdir = rootdir + "/" + folder
            arr1 = os.listdir(newdir)
            for concentration in arr1:
                if ('nm' in concentration) or ('nM' in concentration) or ('Nm' in concentration) or ('NM' in concentration):
                    newdir1 = newdir + "/" + concentration
                    arr2 = os.listdir(newdir1)
                    for CV_file in arr2:
                        if 'CV.txt' in CV_file:
                            count +=1
    num_CVs = count

    #create the CV_data and CV_labels variables
    if include_time is True:
        CV_data = np.zeros(shape=(num_CVs, 1100, 2))
    elif include_time is False:
        CV_data = np.zeros(shape=(num_CVs, 1100, 1))
    else:
        print('include_time is not bool')
        return

    CV_labels = np.zeros(shape=(num_CVs,))
    count = 0 #reset the count
    labels = [10,25,50,75,100]

    for folder in arr:
        if folder != '.DS_Store':
            newdir = rootdir + "/" + folder
            arr1 = os.listdir(newdir)
            for concentration in arr1:
                if ('nm' in concentration) or ('nM' in concentration) or ('Nm' in concentration) or ('NM' in concentration):
                    newdir1 = newdir + "/" + concentration
                    arr2 = os.listdir(newdir1)
                    conc_sp =concentration.split('nm',1)
                    conc = int(conc_sp[0])
                    if catergor is True:
                        for l in range(len(labels)):
                            if conc == int(labels[l]):
                                label = l
                    for CV_file in arr2:
                        if 'CV.txt' in CV_file:
                            filename = newdir1 + "/" + CV_file
                            data = np.genfromtxt(filename)
                            if normalise is True:
                                data[:,2] = data[:,2] / max(data[:,2])

                            #print(data)
                            #print(count)
                            #if (count == 0):
                            #    print(CV_file)
                            #    print(folder)
                            #    print(concentration)

                            if include_time is True:
                                CV_data[count][:,1] = data[:,0]
                            CV_data[count][:,0] = data[:,2]
                            if catergor is True:
                                CV_labels[count] = label
                            else:
                                CV_labels[count] = conc
                            count+=1

    if catergor is True:
        CV_labels= to_categorical(CV_labels,num_classes=5)

    return CV_data, CV_labels

def clean_data(CVs,time=None):
    """

    :param CVs:
    :param also takes an argument as to whether or not to include time

    :return: the data and the concentration labels of the CV data
             the data is also converted to be in the range [-1,1] by dividing by 60
             NB. the labels are catergorical. i.e. [712 x [0 0 1 0 0]] eg (if the label was 50 in this case)
             index[0] = 10, [1] = 25, [2] = 50, [3] = 75, [4] = 100
             also current is the index 0
    """
    #load all the CVs
    #GLOBAL
    shape = CVs.shape

    #fiddle around with CVs to have the data and also the labels (i.e. concentration_as two different tensors
    if time is True:
        data = np.zeros((shape[0],shape[1]-1,shape[2]-1))
    elif time is False:
        data = np.zeros((shape[0],shape[1]-1,1))

    #FOR MILO <3
    milo_labels = np.array([CVs[count][0][0] for count in range(len(CVs))])
    label = [10,25,50,75,100]
    for i in range(len(milo_labels)):
        for index in range(len(label)):
            if milo_labels[i]==label[index]:
                milo_labels[i] = index
        #milo_labels[i] = index for index in range(len(label)) if int(milo_labels[i]) == int(label[index])
    labels = to_categorical(milo_labels,num_classes=5)

    for count, CV in enumerate(CVs):
        if time is True:
            data[count-1][:,1]= CV[1:][:,0]
        data[count-1][:,0]= CV[1:][:,2]

    data = data/60

    return data, labels,

def split_test(data,labels,percentage):
    """
    split the data into testing & training

    :param data:
    :param labels:
    :param percecntage: how much test data you want
    :return: testing and training data
    """
    #ensure different test & training each time
    data,labels = shuffle(data,labels)

    total = len(data)
    index = int(round(total * (percentage / 100)))

    if total == index:
        print('Change the 3rd argument to split the data into test and training')
        return

    test_data = data[:index]
    test_labels = labels[:index]

    train_data = data[index:]
    train_labels = labels[index:]

    return train_data, train_labels,test_data,test_labels

def split_train_val(data,labels,percentage):
    """
    Split the CV data into training and validation data
    :param data: CV data
    :param labels: CV labels
    :param percentage: how much to split the data (will be between 0 and 100) - how much validation you want
    :return:
    """

    total = len(data)
    index = int(round(total * (percentage / 100)))

    if total == index:
        print('Change the 3rd argument to split the data into training and validation')
        return

    val_data = data[:index]
    val_labels = labels[:index]

    train_data = data[index:]
    train_labels = labels[index:]

    return train_data, train_labels, val_data, val_labels



##################################################################################################

# MULTICLASS MODELS

def LSTM_Multi(n_timesteps,rate,time=None,dropout=None,cater=None):
    model = Sequential()

    opt = Adagrad(learning_rate=rate) #how we change the weights
    #opt = tf.keras.optimizers.Adam(learning_rate=rate)
    loss = 'sparse_categorical_crossentropy'
    if cater is True:
        loss = 'categorical_crossentropy' #how wrong we are

    if time is False:
        #model.add(LSTM(64,input_shape = (n_timesteps,1),return_sequences= True))
        model.add(LSTM(64, input_shape=(n_timesteps, 1)))
    if time is True:
        model.add(LSTM(64,input_shape = (n_timesteps,2)))
    #model.add(Dense(100,activation='relu'))
    #model.add(BatchNormalization())
    if dropout is True:
        model.add(layers.Dropout(0.5))

    model.add(LayerNormalization())

#    model.add(LSTM(64,return_sequences=True))
#    model.add(LSTM(32,return_sequences=True))
    #model.add(LSTM(32))
    model.add(Dense(5, activation='softmax')) #final layer with 5 output classes
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model

def RNN(n_timesteps,time=None,dropout=None): #this generally is not fab because of vanishing grad prob
    model=Sequential()
    if time is False:
        model.add(SimpleRNN(64,input_shape = (n_timesteps,1),return_sequences=True))
    if time is True:
        model.add(SimpleRNN(64,input_shape = (n_timesteps,2),return_sequences=True))

    if dropout is True:
        model.add(layers.Dropout(0.5))

    model.add(SimpleRNN(32))
    model.add(Dense(5, activation='softmax')) #final layer with 5 output classes
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model

def GRU_model(n_timesteps,time=None,dropout=None):
    model = Sequential()
    if time is False:
        model.add(GRU(64,input_shape = (n_timesteps,1)))
    if time is True:
        model.add(GRU(64,input_shape = (n_timesteps,2)))
    #model.add(Dense(100,activation='relu'))
    if dropout is True:
        model.add(layers.Dropout(0.5))


    model.add(Dense(5, activation='softmax')) #final layer with 5 output classes
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model

# REGRESSION MODELS
def LSTM_reg(n_timesteps,rate,time=None,dropout=None):
    model = Sequential()
    opt = Adagrad(learning_rate=rate) #how we change the weights
    #opt = 'rmsprop'
    loss = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")

    if time is False:
        model.add(LSTM(64,input_shape = (n_timesteps,1),activation='tanh'))
    if time is True:
        model.add(LSTM(64,input_shape = (n_timesteps,2)))

    model.add(Dense(1)) #there is no activation function here as we do not want to transform the data
    model.compile(loss=loss, optimizer=opt, metrics=['mae'])

    model.summary()

    return model

# BINARY MODELS
def LSTM_bin(n_timesteps,rate,time=None,dropout=None):
    model = Sequential()

    opt = RMSprop(learning_rate=rate)  # how we change the weights
    # opt = tf.keras.optimizers.Adam(learning_rate=rate)

    loss = 'binary_crossentropy'  # how wrong we are

    if time is False:
        # model.add(LSTM(64,input_shape = (n_timesteps,1),return_sequences= True))
        model.add(LSTM(64, input_shape=(n_timesteps, 1)))
    if time is True:
        model.add(LSTM(64, input_shape=(n_timesteps, 2)))
    # model.add(Dense(100,activation='relu'))
    # model.add(BatchNormalization())
    if dropout is True:
        model.add(layers.Dropout(0.5))

    #    model.add(LSTM(64,return_sequences=True))
    #    model.add(LSTM(32,return_sequences=True))
    # model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))  # final layer with 1 output classes
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model

# MODEL FUNCTIONS

def dif_acc_metric(predictions,test_lab,class_dif):
    total = len(predictions)
    correct = 0
    for p in range(total):#go through each prediction and calculate how many times
        conc = np.where(predictions[p] == 1.)[0]
        label = np.where(test_lab[p]==1.)[0]
        #go through the class difference
        for i in range(class_dif+1):
            if (conc + i == label) or (conc - i) == label:
                correct+=1
                break
    new_metric = correct/total

    return new_metric

def predict_test(model,test_data,test_lab,classifier = None,confusion_mat=True):
    pred = model.predict(test_data)
    if classifier == 'multi':
        predictions = np.zeros((len(pred), 5))
        for count, i in enumerate(pred):
            max_index = np.where(i == max(i))
            predictions[count][max_index] = 1
    elif classifier == 'binary':
        predictions = np.zeros((len(pred), 1))
        for count,i in enumerate(pred):
            if i > 0.5:
                predictions[count] = 1

    if confusion_mat is True:
        if classifier == 'multi':
            # visulize the predictions in a confusion matrix
            plt.clf()
            cm = confusion_matrix(test_lab.argmax(axis=1), predictions.argmax(axis=1), normalize='true')#,labels=["10 nm","25 nm","50 nm","75 nm","100 nm"])
            print('\n', cm)
            cm1 = confusion_matrix(test_lab.argmax(axis=1), predictions.argmax(axis=1))
            print('\n', cm1)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.imshow(cm)
            plt.colorbar(label='Proportion Correct')
            plt.show()
        elif classifier == 'binary':
            plt.clf()
            cm = confusion_matrix(test_lab, predictions, normalize='true')
            print('\n', cm)
            cm1 = confusion_matrix(test_lab, predictions)
            print('\n', cm1)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.imshow(cm)
            plt.colorbar()
            plt.show()
        else:
            print('Please specify which type of classifier you are performing to visualize the confusion matrix. \n Accepted arguments include : \'binary\' or \'multi\'')
    return pred, predictions

def save_model(model,model_type,epoch,dropout = bool):
    if isinstance(model_type,str) is False:
        print('model_type Argument must be a string')
        return
    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime("%m.%d.%Y_%H.%M", named_tuple)
    if dropout is True:
        model.save("/Users/nathanrobins/Documents/MSc_Neurotech/MSc_Project/Models/" + model_type +"_"+ str(epoch) + "epochs_" + str(
        time_string) + "dropout" + ".h5")
    else:
        model.save("/Users/nathanrobins/Documents/MSc_Neurotech/MSc_Project/Models/" + model_type + "_" + str(
            epoch) + "epochs_" + str(time_string) + ".h5")
    return
##################################################################################################

#PLOT

def ML_plot(history,type):
    if type == 'multi' or type == 'binary':
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        epochs = range(1, len(loss) + 1)

        plt.subplot(211)
        plt.plot(epochs, loss, 'b--', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        #plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(212)
        plt.plot(epochs, acc, 'r--', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()

    elif type == 'regression':
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        mae = history.history['mae']
        val_mae = history.history['val_mae']

        epochs = range(1, len(loss) + 1)

        plt.subplot(211)
        plt.plot(epochs, loss, 'b--', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        # plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(212)
        plt.plot(epochs, mae, 'r--', label='Training MAE')
        plt.plot(epochs, val_mae, 'r', label='Validation MAE')
        plt.title('Training and validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.legend()

        plt.show()

    else:
        print('Incorrect Second Argument \n Try "regression", "binary" or "multi"')

    return


##################################################################################################

#PERFORM PREDICTIONS

def load_m(model_name):
    modeldir = "/Users/nathanrobins/Documents/MSc_Neurotech/MSc_Project/Models/"
    model = load_model(modeldir + model_name)  # whatever model we want
    return model

def load_test_data(dir_oi, include_time=bool):
    rootdir = "/Users/nathanrobins/Documents/MSc_Neurotech/MSc_Project/Data/unseen_data/"
    dir = rootdir + dir_oi  # this is the directory that has all of the data we are interested in

    # initially go through the directory and find out how many CV files there are:
    CV_count = 0
    arr = os.listdir(dir)
    for CV_file in arr:
        if 'CV.txt' in CV_file:
            CV_count += 1

    # create the empty data variable to be filled with the data in the files
    if include_time is True:
        data = np.zeros(shape=(CV_count, 1100, 2))
    elif include_time is False:
        data = np.zeros(shape=(CV_count, 1100, 1))
    else:
        print('include_time is not bool')
        return

    count = 0
    # store the data
    for CV_file in arr:
        if 'CV.txt' in CV_file:
            filename = dir + "/" + CV_file
            CV = np.genfromtxt(filename)
            if include_time is True:
                data[count][:, 1] = CV[:, 0]
            data[count][:, 0] = CV[:, 2]

            count += 1

    return data

def predictions(model, data, model_type):
    predictions = model.predict(data)
    if model_type == 'multiclass':
        pred = [np.argmax(predict) for predict in predictions]
        classes = [10, 25, 50, 75, 100]
        for count, p in enumerate(pred):
            for c in classes:
                if int(p) == int(classes.index(c)):
                    pred[count] = c

    # plot
    plt.plot(pred,'o',markersize=6)
    plt.xlabel('Time (mins)')
    plt.ylabel('Concentration (nm)')
    plt.grid()
    #plt.show()
    plt.clf()
    return pred

def multi_weighted_predictions(model,data,colour,title,plot=True):
    """
    This function returns single values of concentration, based on the weighted probabilities in the class prediction of the
    multiclass classifier
    """

    font = {  # 'family' : 'normal',
        # 'weight' : 'bold',
        'size': 20}

    matplotlib.rc('font', **font)

    probabilities = model.predict(data)
    classes = [10,25,50,75,100]
    predictions = np.zeros(len(data))
    for count,prob in enumerate(probabilities): #go through every probability
        idx = int(np.where(prob == max(prob))[0]) #this is the index of the most likely class
        if idx == 0: #the first class
            fraction = prob[idx+1]/(prob[idx] + prob[idx+1])
            value = classes[idx] + ((classes[idx+1]-classes[idx])*fraction)

        elif idx == 4: #final class
            fraction = prob[idx-1] / (prob[idx] + prob[idx - 1])
            value = classes[idx] - ((classes[idx] - classes[idx - 1]) * fraction)

        elif prob[idx + 1] > prob[idx - 1]: #if the class above is more likely than the class below
            fraction = prob[idx+1] / (prob[idx] + prob[idx + 1])
            value = classes[idx] + ((classes[idx + 1] - classes[idx]) * fraction)

        elif prob[idx + 1] < prob[idx - 1]:# if the class below is more likely than the class above
            fraction = prob[idx-1] / (prob[idx] + prob[idx - 1])
            value = classes[idx] - ((classes[idx] - classes[idx-1]) * fraction)

        else: #if the probabilities the same
            value = classes[idx]

        predictions[count] = value
    if plot is True:
        plt.plot(predictions,colour,markersize=6)
        plt.xlabel('Time (mins)')
        plt.ylabel('Concentration (nm)')
        plt.grid()
        plt.title(title)
        plt.show()

    return predictions