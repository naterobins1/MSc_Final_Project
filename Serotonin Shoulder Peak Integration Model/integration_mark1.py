"""

Initial analysis of CV data (17.06.20)

Written for python 3

"""

#imports
from Integration.inception import inception
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from Integration.int_functions import some_int

# here we have a variable CVs that contains all CV for all electrodes @ all concentrations
# but we want to look at specific electrodes and concentrations

global_ = True
indv_concentration = False
indv_electrode = False

if global_ is True:
    def findconc_charge(inflate=False):
        CVs = inception("/Users/nathanrobins/Documents/MSc_Neurotech/MSc_Project/Data", "In_vitro_basal_5-HT",'global')
        #create these variables for ploting of the relationship between concentration & charge
        concs = np.zeros(shape =(len(CVs),))
        charges = np.zeros(shape=(len(CVs),))

        #go through each CV file & calculate the charge
        for i in range(len(CVs)):
            CVs[i][0][1] = some_int(CVs[i])  #store the charge in in the CVs array
            concs[i] = CVs[i][0][0]
            charges[i] = CVs[i][0][1]

        if inflate is True:
            count = 0
            for charge in charges:
                print(charge)
                if charge < 0.009:
                    count+=1
            concs = np.zeros(shape=(count,))
            charges = np.zeros(shape=(count,))
            count = 0
            for i in range(len(CVs)):
                charge = some_int(CVs[i])
                if charge < 0.009:
                    concs[count] = CVs[i][0][0]
                    charges[count] = charge
                    count +=1

        return concs, charges

    #concs, charges = findconc_charge()

    concs1, charges1 = findconc_charge(inflate=True)

    def int_analysis(concs,charges):
        print('Pearson\'s r: ',scipy.stats.pearsonr(concs, charges))
        print('\n',scipy.stats.spearmanr(concs, charges))
        print('\n', scipy.stats.kendalltau(concs, charges))
        #now we want to look at the relationship between the charge & the concentration for all in vitro files
        plt.figure(1)
        plt.plot(concs,charges,'o')

        plt.xlabel('Concentration (nm)')
        plt.ylabel('Integrated area (unitless)')
        #plt.title('The relationship between serotonin concentration and integrated area calculated using Model 1')

        plt.grid()
        plt.show()
        plt.clf()
        return

    #int_analysis(concs,charges)
    int_analysis(concs1,charges1)



elif indv_concentration is True:
    CVs = inception("/Users/nathanrobins/Documents/MSc_Neurotech/MSc_Project/Data", "In_vitro_basal_5-HT",'global')
    #look at specific concentrations
    concentrations = [10,25,50,75,100] #nm
    for counter,concentration in enumerate(concentrations,1): #each different concentration
        charges = []
        for CV in range(len(CVs)): #go through each CV and see if they are of the concentration of interest
            if CVs[CV][0][0] == concentration:
                charge = some_int(CVs[CV])
                charges = np.append(charges,charge) #create a short list for each concentration
        #charges.append(chargess) #then append this short list to the larger list to get 5 arrays within the list

    #visualise the concentration data & look at the relationship across concentrations
    #they should be clumped together - ish
        if counter == 1:
            ax1 = plt.subplot(len(concentrations),1,counter)
            ax1.plot(charges)
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.title('Relationship of charge within indv concentrations')
        else:
            ax1 = plt.subplot(len(concentrations), 1, counter,sharex=ax1,sharey=ax1)
            plt.setp(ax1.get_xticklabels(), visible=False)
            ax1.plot(charges)

    plt.show()



elif indv_electrode is True:
    CVs = inception("/Users/nathanrobins/Documents/MSc_Neurotech/MSc_Project/Data", "In_vitro_basal_5-HT",'electrode')
    print(CVs[0][0][0][0])
    #plot each different electrode
    #go through each electrode
    for counter,electrode in enumerate(CVs,1):
        concs = np.zeros(shape=(len(electrode),))
        charges = np.zeros(shape=(len(electrode),))
        #print(len(electrode))
        for CV in range(len(electrode)): #go through each CV in each electrode and calculate the charge
            #counter-1 = the electrode
            CVs[counter-1][CV][0][1] = some_int(electrode[CV])  # store the charge in in the CVs array
            concs[CV] = CVs[counter-1][CV][0][0]
            charges[CV] = CVs[counter-1][CV][0][1]

        if counter == 1:
            ax1 = plt.subplot(5, 2, counter)
            ax1.plot(concs, charges,'o')
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.title('Relationship of charge within indv electrodes')
        else:
            ax1 = plt.subplot(5, 2, counter, sharex=ax1, sharey=ax1)
            plt.setp(ax1.get_xticklabels(), visible=False)
            ax1.plot(concs,charges,'o')

    plt.show()


else:
    print('Error - neither global nor local specified')