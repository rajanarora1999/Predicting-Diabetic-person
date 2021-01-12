#import reqd libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from sklearn.naive_bayes import MultinomialNB
def prior_prob(y_train,label):
    
    total_examples = y_train.shape[0]
    class_examples = np.sum(y_train==label)
    
    return (class_examples)/float(total_examples)

def cond_prob(x_train,y_train,feature_col,feature_val,label):
    
    x_filtered = x_train[y_train==label]
    numerator = np.sum(x_filtered[:,feature_col]==feature_val)
    denominator = np.sum(y_train==label)
    
    return numerator/float(denominator)

def predict(x_train,y_train,xtest):
    """Xtest is a single testing point, n features"""
    
    classes = np.unique(y_train)
    n_features = x_train.shape[1]
    post_probs = [] # List of prob for all classes and given a single testing point
    #Compute Posterior for each class
    for label in classes:
        
        #Post_c = likelihood*prior
        likelihood = 1.0
        for f in range(n_features):
            cond = cond_prob(x_train,y_train,f,xtest[f],label)
            likelihood *= cond 
            
        prior = prior_prob(y_train,label)
        post = likelihood*prior
        post_probs.append(post)
        
    pred = np.argmax(post_probs)
    return pred

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
    #train the model
    outcomes=[]
    ytrain=ytrain.reshape(-1,)
    for i in range(xtest.shape[0]):
        outcomes.append(predict(xtrain,ytrain,xtest[i]))
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
