"""

Call the functions from ML_functions.py to actually perform ML

"""

import numpy as np

#functions
from ML.ML_functions import split_train_val, split_test, load_all_data
from ML.ML_functions import LSTM_Multi, RNN, GRU_model
from ML.ML_functions import ML_plot, predict_test , save_model
from Visualisation.visualisation import number_concentrations

drop = False


#are we going to include time as an input for the network to learn on?
epoch = int(input("How many epochs for training?"))
save = (input("Do you want to save the model? (Y or N)"))
#cater = input("Catergorical labels? (Y or N)")

if save == "Y":
    save = True
elif save == "N":
    save = False

incl_time = False
cater = "Y"
if cater == "Y":
    cater = True
elif cater == "N":
    cater = False

CV_data,CV_lab =load_all_data(include_time=incl_time,catergor=cater,normalise=False)
if 1 ==1:
    if cater is False:
        conc = [10, 25, 50, 75, 100]
        for number, label in enumerate(CV_lab):
            for count, c in enumerate(conc):
                if int(c) == int(label):
                    CV_lab[number] = count


#split into training/val & testing
CV_data,CV_lab,test_data,test_lab = split_test(CV_data,CV_lab,10)

#split into training and validation
train_data, train_labels, val_data, val_labels = split_train_val(CV_data,CV_lab,15)

model_choice = 'LSTM'

timesteps = len(CV_data[0])
no_CVs = len(CV_data)
if model_choice =='LSTM':
    print('\nModel Choice: Long Short-Term Memory\n')
    no_conc = number_concentrations(train_labels,catergorical=cater)
    print(no_conc)

    rate = 0.03
    class_weight={0: no_CVs/no_conc[0],
                  1: no_CVs/no_conc[1],
                  2: no_CVs/no_conc[2],
                  3: no_CVs/no_conc[3],
                  4: no_CVs/no_conc[4]}
    #print(class_weight)
    #callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, baseline=1)

    model = LSTM_Multi(timesteps,rate,time=incl_time,dropout=drop,cater=cater)
    history = model.fit(train_data, train_labels,
                    epochs=epoch,
                    batch_size=10,
                    validation_data=(val_data, val_labels),
                    class_weight=class_weight
                       )

    if save is True:
        #save the model, named by when it was made
        save_model(model,'multi',epoch,dropout=drop)

    ML_plot(history,type='multi')

    # test the model
    print()
    print('Testing the model...')
    results = model.evaluate(test_data, test_lab)
    print(results)

    # lets see what the model predicts
    probabilities, predictions = predict_test(model,test_data,test_lab,classifier='multi',confusion_mat=False)
    #print(predictions)

