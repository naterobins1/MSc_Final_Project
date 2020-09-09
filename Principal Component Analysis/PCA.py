"""

https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/

"""

#imports
from ML.ML_functions import load_all_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import math

def PCA_analysis(data,labels,n_components,principle_components_to_view,plot=False):
    X_train, X_test, y_train, y_test = train_test_split(data,labels, test_size=0.2, random_state=0)

    X_train.resize((X_train.shape[0],X_train.shape[1]))
    X_test.resize((X_test.shape[0],X_test.shape[1]))

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    if n_components ==0:
        pca = PCA()
    else:
        pca = PCA(n_components=n_components)
    princ_comp = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    targets = pd.DataFrame(data = y_train,columns=['Concentrations'])
    if n_comp ==3:
        principalDf =pd.DataFrame(data = princ_comp,columns=['PC 1','PC 2',"PC 3"])
    elif n_comp ==2:
        principalDf =pd.DataFrame(data = princ_comp,columns=['PC 1','PC 2'])
    else:
        principalDf =pd.DataFrame(data = princ_comp)

    finalDf = pd.concat([principalDf,targets ], axis=1)

    explained_variance = pca.explained_variance_ratio_


    if plot is True:
        if principle_components_to_view == 'all':
            plt.plot(explained_variance)
        else:
            plt.plot(explained_variance[0:principle_components_to_view+1])
        plt.xlabel('Principal Component')
        plt.ylabel('Contribution to Variance')

        #plt.show()

        if principle_components_to_view == 'all':
            print('\nContribution to variance of the PCs: ', explained_variance)
            print('Total contribution: ', sum(explained_variance))
        else:
            print(explained_variance[0:principle_components_to_view])
            print(sum(explained_variance[0:principle_components_to_view]))

    return pca,finalDf, explained_variance

#load the dtaa
CV_data,CV_lab = load_all_data(include_time=False,catergor=False)

n_comp = 3 #number of components to peform analysis on
principle_components_to_view = 'all' #no_components to view
pca,PCs,explained_variance = PCA_analysis(CV_data,CV_lab,n_comp,principle_components_to_view,plot=True)

font = {#'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)

plt.clf()
if n_comp == 3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.scatter(PCs["PC 1"],PCs["PC 2"],PCs["PC 3"],c=PCs['Concentrations'],cmap='rainbow')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    plt.colorbar(surf,ax=ax,label='Concentration (nm)')

    #ax.legend(['10nm','25nm','50nm','75nm','100nm'])
    #ax.set_title('Principal Component Analysis')
    ax.grid()
elif n_comp ==2:
    plt.scatter(PCs["PC 1"],PCs["PC 2"],c=PCs['Concentrations'],label = PCs['Concentrations'],cmap='rainbow')
    #plt.title('Principal Component Analysis')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.colorbar(label='Concentration (nm)')
    #plt.legend()
    plt.grid()
#plt.show()

plt.clf()
map= pd.DataFrame(pca.components_)
plt.figure()


if n_comp == 3:
    ax = sns.heatmap(map,cmap='twilight',yticklabels=['PC 1','PC 2','PC 3'],cbar_kws={'label': 'Correlation (unitless)'})
elif n_comp == 2:
    ax = sns.heatmap(map,cmap='twilight',yticklabels=['PC 1','PC 2'],cbar_kws={'label': 'Correlation (unitless)'})


#plt.title('Correlation between the data points and the principal components')
plt.xlabel('Data Points')
#plt.show()