#import reqd libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
#distance function to be used by knn
def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))
#knn algorithm( x->training data, y->labels of training data,qp->query point, k->number of neighbours in consideration)
def knn(x,y,qp,k=5):
	#vals list to store distances and their labels
    vals=[]
    #iterate over the training data and append distance and label for each point
    for i in range(x.shape[0]):
        vals.append((dist(x[i],qp),y[i]))
    #sort the list so that k nearest can be taken
    vals=sorted(vals)
    #take first k nearest
    vals=vals[:k]
    #convert list into numpy array
    vals=np.array(vals)
    # now take how many unique labels are there in k nearest(use only 1st column for labels)
    #and also return their count
    #new vals is a list with two tupls
    #first tuple has labels and second has their freq.
    new_vals=np.unique(vals[:,1],return_counts=True)
    #take the index of max freq label
    max_freq_index=new_vals[1].argmax()
    #return the label with max freq
    return new_vals[0][max_freq_index]



if __name__=='__main__':
	print("This model uses KNN Algorithm to predict whether a person has diabetes or not based on given features")
	print("Diabetes Data set consisting of following features :'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'")
	#convert given data set to a pandas dataframe where xtrain is training data and y train is its label
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
	outcomes=[knn(xtrain,ytrain,xtest[i]) for i in range(xtest.shape[0])]
	
	#list to be used for table
	final_answer=[]
	for i in range(xtest.shape[0]):
		row=xtest[i].tolist()
		if outcomes[i]==1:
			row.append("Diabetic")
		else :
			row.append('Not Diabetic')
		final_answer.append(tuple(row))
	# ptint table 
	#tabulate takes list of tuples for rows and a list of headers 
	print(tabulate(final_answer,headers=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']))
    