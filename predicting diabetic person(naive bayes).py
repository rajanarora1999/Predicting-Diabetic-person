#import reqd libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from sklearn.naive_bayes import MultinomialNB
if __name__=='__main__':
    xtrain=pd.read_csv('Diabetes_XTrain.csv')
    ytrain=pd.read_csv('Diabetes_Ytrain.csv')
	#convert them to numpy array so that slicing is easy
    xtrain=np.array(xtrain)
    ytrain=np.array(ytrain)
	#make dataframe of testing data and convert into numpy array
    xtest=pd.read_csv('Diabetes_Xtest.csv')
    xtest=np.array(xtest)
	# to check the shape of testing data
	#xtest.shape
	#store the outcomes in a list so that these can be added to table
    #multinomial naive bayes
    mnb=MultinomialNB()
    #train the model
    mnb.fit(xtrain,ytrain)
    outcomes=mnb.predict(xtest)
    final_answer=[]
    for i in range(xtest.shape[0]):
        row=xtest[i].tolist()
        if outcomes[i]==1:
            row.append("Diabetic")
        else :
            row.append('Not Diabetic')
        final_answer.append(tuple(row))
        # print table 
        #tabulate takes list of tuples for rows and a list of headers 
    print(tabulate(final_answer,headers=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']))
