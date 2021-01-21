#import reqd libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
ytrain[ytrain==0]=-1

#SVM Code with only 2 classes i.e -1 & 1
class SVM:
    def __init__(self,C=1.0):
        #C->hyperparameter
        self.C = C
        #W-> weights
        self.W = 0
        #b->bias
        self.b = 0
        
    def hingeLoss(self,W,b,X,Y):
        loss  = 0.0
        
        loss += .5*np.dot(W,W.T)
        
        m = X.shape[0]
        
        for i in range(m):
            ti = Y[i]*(np.dot(W,X[i].T)+b)
            loss += self.C *max(0,(1-ti))
            
        return loss[0][0]
    
    def fit(self,X,Y,batch_size=100,learning_rate=0.0000001,maxItr=1000):
        
        no_of_features = X.shape[1]
        no_of_samples = X.shape[0]
        
        n = learning_rate
        c = self.C
        
        #Init the model parameters
        W = np.zeros((1,no_of_features))
        bias = 0
        
        #Initial Loss
        
        #Training from here...
        # Weight and Bias update rule!
        losses = []
        
        for i in range(maxItr):
            #Training Loop
            
            l = self.hingeLoss(W,bias,X,Y)
            losses.append(l)
            ids = np.arange(no_of_samples)
            np.random.shuffle(ids)
            
            #Batch Gradient Descent(Paper) with random shuffling
            for batch_start in range(0,no_of_samples,batch_size):
                #Assume 0 gradient for the batch
                gradw = 0
                gradb = 0
                
                #Iterate over all examples in the mini batch
                for j in range(batch_start,batch_start+batch_size):
                    if j<no_of_samples:
                        i = ids[j]
                        ti =  Y[i]*(np.dot(W,X[i].T)+bias)
                        
                        if ti>1:
                            gradw += 0
                            gradb += 0
                        else:
                            gradw += c*Y[i]*X[i]
                            gradb += c*Y[i]
                            
                #Gradient for the batch is ready! Update W,B
                W = W - n*W + n*gradw
                bias = bias + n*gradb
                
        
        self.W = W
        self.b = bias
        return W,bias,losses

mysvm=SVM()
w,b,loss=mysvm.fit(xtrain,ytrain)
def binaryPredict(x,w,b):
    z=np.dot(x,w.T)+b
    if z>=0:
        return "Diabetic"
    else :
        return "Not Diabetic"

outcomes=[]
for i in range(xtest.shape[0]):
    outcomes.append(binaryPredict(xtest[i],w,b))
print(outcomes)