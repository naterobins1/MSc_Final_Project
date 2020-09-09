"""

Code built on top of load_data_test.py trying to obtain all data from all CV files

Kinda reminds me of inception... always one layer deeper

Written for python 3

"""

#imports
import numpy as np
import os
from Integration.add_conc import getdata_addconc
from keras.utils.np_utils import to_categorical


#in_vitro = "In_vitro_basal_5-HT"
#in_vivo = "In_vivo"

#rootdir = "/Users/nathanrobins/Documents/MSc_Neurotech/MSc_Project/Data"

def inception(rootdir,vivo_vitro,seperate_by): #argument seperate_by will make different arrays depending on
                                                #concentration or electrode. Optional argument however
#the root directory to access different files

    arr = os.listdir(rootdir)

    #create an empty array all_CV to add the CV arrays to at the end
    #660 .txt files so far
          #each one has 1100 data points & then one for concentration storage

    if seperate_by =='electrode':
        print('Seperating CVs by electrode')
        first = True
        for i in range(len(arr)):  # go through in vitro vs in vivo
            if arr[i] == vivo_vitro:  # only look at the files in vitro
                newdir = rootdir + "/" + vivo_vitro
                arr1 = os.listdir(newdir)
                for n in range(len(arr1)):
                    if len(arr1[n]) == 6:  # file of interest are only 6 digits long
                        newdir1 = newdir + "/" + arr1[n]
                        # newdir1 is now all of the folders with something like 110818

                        # now need to go one level lower
                        # electrode layer
                        arr2 = os.listdir(newdir1)
                        for x in range(len(arr2)):
                            if len(arr2[x]) == 10:  # again this works but it is a little bit hacky
                                #print(arr2[x])
                                # this ensures that we are looking at the electrodex folders
                                electrode_l = 0
                                # if we want to seperate the CVs by electrode

                                newdir2 = newdir1 + "/" + arr2[x]

                                # inception further
                                arr3 = os.listdir(newdir2)

                                for y in range(len(arr3)):  # goes through every concentration
                                    if 'nm' in arr3[y]:  # this if loop ensires only the conc files are looked at
                                        newdir3 = newdir2 + "/" + arr3[y]

                                        # find the concentration value by taking all digits to the left of the nm
                                        #split = arr3[y].split('nm')
                                        #conc = int(split[0])

                                        # inception
                                        arr4 = os.listdir(newdir3)

                                        for count in range(len(arr4)):
                                            if '.txt' in arr4[count]:
                                                # here I am now dealing with individual CV files
                                                #filename = newdir3 + "/" + arr4[count]
                                                #CV = getdata_addconc(filename, conc)
                                                #CVs[count] = CV
                                                electrode_l +=1 #initally only find out how many CVs are there for

                                                                #this electrode

                                electrode_CV = np.zeros(shape=(electrode_l, 1101, 3))
                                count = 0
                                for y in range(len(arr3)):  # goes through every concentration
                                    if 'nm' in arr3[y]:  # this if loop ensires only the conc files are looked at
                                        newdir3 = newdir2 + "/" + arr3[y]

                                        # find the concentration value by taking all digits to the left of the nm
                                        split = arr3[y].split('nm')
                                        conc = int(split[0])

                                        # inception
                                        arr4 = os.listdir(newdir3)
                                        for c in range(len(arr4)):
                                            if '.txt' in arr4[c]:
                                                # here I am now dealing with individual CV files
                                                filename = newdir3 + "/" + arr4[c]
                                                CV = getdata_addconc(filename,conc)
                                                CV.tolist()
                                                electrode_CV[count] = CV
                                                count =count+1
                                                #we have now created an electrode_CV for each electrode

                                #at this level the electrode_CV var is all CVs per electrode
                                #add electrode_CV to the global CVs var
                                if first is True:
                                    electrode_CV = electrode_CV.tolist()
                                    CVs = [electrode_CV]
                                    first = False

                                else:
                                    electrode_CV = electrode_CV.tolist()
                                    CVs.append(electrode_CV)



##################################################################################################################

    elif seperate_by =='global':
        print('Global Analysis Occuring...')
        CVs = np.zeros(shape=(712, 1101, 3))
        count = 0
        for i in range(len(arr)): #go through in vitro vs in vivo
            if arr[i] == vivo_vitro: #only look at the files in vitro
                newdir = rootdir + "/" + vivo_vitro
                arr1 = os.listdir(newdir)
                for n in range(len(arr1)):
                    if len(arr1[n]) ==6: #file of interest are only 6 digits long
                        newdir1 = newdir + "/" + arr1[n]
                        #newdir1 is now all of the folders with something like 110818

                        #now need to go one level lower
                        #electrode layer
                        arr2 = os.listdir(newdir1)
                        for x in range(len(arr2)):
                            if len(arr2[x]) ==10: #again this works but it is a little bit hacky
                                # this ensures that we are looking at the electrodex folders

                                #if we want to seperate the CVs by electrode




                                newdir2 = newdir1 + "/" + arr2[x]

                                #inception further
                                arr3 = os.listdir(newdir2)

                                for y in range(len(arr3)): #goes through every concentration
                                    if 'nm' in arr3[y]: #this if loop ensires only the conc files are looked at
                                        newdir3 = newdir2 + "/" + arr3[y]

                                        # find the concentration value by taking all digits to the left of the nm
                                        split = arr3[y].split('nm')
                                        conc = int(split[0])

                                        # inception
                                        arr4 = os.listdir(newdir3)

                                        for c in range(len(arr4)):
                                            if '.txt' in arr4[c]:
                                                # here I am now dealing with individual CV files
                                                filename = newdir3 + "/" + arr4[c]
                                                CV = getdata_addconc(filename,conc)
                                                CVs[count] = CV
                                                count = count + 1

    return CVs


def load_all_data(include_time = None):
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
    num_CVs = count #2020

    #create the CV_data and CV_labels variables
    if include_time is True:
        CV_data = np.zeros(shape=(num_CVs, 1100, 2))
    elif include_time is False:
        CV_data = np.zeros(shape=(num_CVs, 1100, 1))

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
                    for l in range(len(labels)):
                        if conc == int(labels[l]):
                            label = l
                    for CV_file in arr2:
                        if 'CV.txt' in CV_file:
                            filename = newdir1 + "/" + CV_file
                            data = np.genfromtxt(filename)
                            #print(data)
                            #print(count)
                            #if (count == 0):
                            #    print(CV_file)
                            #    print(folder)
                            #    print(concentration)

                            if include_time is True:
                                CV_data[count][:,1] = data[:,0]
                            CV_data[count][:,0] = data[:,2]
                            CV_labels[count] = label
                            count+=1

    CV_data=CV_data/60
    CV_labels= to_categorical(CV_labels,num_classes=5)

    return CV_data, CV_labels
