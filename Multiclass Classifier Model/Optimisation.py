""""

Optimisise different hyperparameters

"""

#imports
from ML.ML_functions import split_train_val, split_test, load_all_data, predict_test
from ML.ML_functions import LSTM_Multi
from Visualisation.visualisation import number_concentrations

import numpy as np

def perform_multi(rate,epoch,batch):
    CV_data,CV_lab =load_all_data(include_time=False)

    #split into training/val & testing
    CV_data,CV_lab,test_data,test_lab = split_test(CV_data,CV_lab,10)

    #split into training and validation
    train_data, train_labels, val_data, val_labels = split_train_val(CV_data,CV_lab,15)

    timesteps = len(CV_data[0])
    no_CVs = len(CV_data)

    no_conc = number_concentrations(train_labels,catergorical=True)
    class_weight={0: no_CVs/no_conc[0],
                  1: no_CVs/no_conc[1],
                  2: no_CVs/no_conc[2],
                  3: no_CVs/no_conc[3],
                  4: no_CVs/no_conc[4]}
    #callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, baseline=1)

    model = LSTM_Multi(timesteps,rate,time=False,dropout=False,cater=True)
    history = model.fit(train_data, train_labels,
                    epochs=epoch,
                    batch_size=batch,
                    validation_data=(val_data, val_labels),
                    class_weight=class_weight)

    results = model.evaluate(test_data, test_lab)

    return results

rate_results = np.zeros((19,2))
rates_to_test = np.zeros(19) # 0.01 to 0.10 and then 0.10 to 1.0
for count,i in enumerate(range(1,11)):
    rates_to_test[count]=i/100
for count,i in enumerate(range(2,11)):
    rates_to_test[count+10] =i/10

batch_results = np.zeros((3,2))
#batch_to_test = [10,100,250,500,750,1100]
batch_to_test = [5,8,10]

epoch = 100


for count,batch in enumerate(batch_to_test):
    print('\n###########################################################################################################')
    print('Batch Size = ', batch)
    batch_results[count]=perform_multi(0.03,epoch,batch)

with open("/Users/nathanrobins/Documents/MSc_Neurotech/MSc_Project/Data/optimisation/multiclass_batch_size_5-8-10_attempt_2.txt", "w+") as f:
    f.write(str(batch_results))