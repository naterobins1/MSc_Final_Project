"""

This script is where we test the unseen data and try to get some results

Also fiddling around with the test data predictions

"""

#imports
from ML.ML_functions import load_m, load_test_data, predictions
from ML.ML_functions import load_all_data,split_test, predict_test, dif_acc_metric, multi_weighted_predictions

#load the model
model = load_m('multi_500epochs_09.05.2020_06.30.h5')

#load the data
control_data = load_test_data('LPS+FLUOX/control',include_time=False)
LPS_data = load_test_data('LPS+FLUOX/lps',include_time=False)
fluox_data = load_test_data('LPS+FLUOX/fluox',include_time=False)
osc_12_data = load_test_data('oscillations/basal_12hr',include_time=False)
osc_3_data = load_test_data('oscillations/basal_3hr',include_time=False)


#CV_data,CV_lab =load_all_data(include_time=False,catergor=True,normalise=False)
#CV_data,CV_lab,test_data,test_lab = split_test(CV_data,CV_lab,10)
#results = model.evaluate(test_data, test_lab)
#print(results)

#probabilities, predictions = predict_test(model,test_data,test_lab,classifier='multi',confusion_mat=True)
#print('Predictions:\n',predictions, '\n\nTest Labels:\n',test_lab,'\n\n','Probabilites:\n',probabilities)

#new_met = dif_acc_metric(predictions,test_lab,1)
#print(new_met)

########################################################################################################################

#test the model on the data
def test_model(model,data,colour,title):
    #print(str(data),'\n')
    predict = predictions(model,data,'multiclass')
    weighted_pred = multi_weighted_predictions(model,data,colour,title,plot=True)
    #data_probs = model.predict(data)
    #print(data_probs,'\n')
    #print(weighted_pred)
    return

test_model(model,control_data,'-','Control')
test_model(model,LPS_data,'c-','LPS')
test_model(model,fluox_data,'r-','Fluoxetine')
#test_model(model,osc_3_data)
#test_model(model,osc_12_data)

