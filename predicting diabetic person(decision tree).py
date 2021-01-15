#import reqd libraries
import pandas as pd
import numpy as np

df=pd.read_csv('Diabetes_Xtrain.csv')
df2=pd.read_csv('Diabetes_Ytrain.csv')
#join both the dataframes to form a single dataframe
df=pd.concat([df,df2],axis=1)

#segregate input and output columns
input_cols=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
out_cols=["Outcome"]

# a function to calculate the entropy 
def entropy(col):
    #unique returns the number of unique values as well as their frequencies
    counts=np.unique(col,return_counts=True)
    N=float(col.shape[0])
    ent=0.0
    #we traverse only frequencies to get probability
    for ix in counts[1]:
        P=ix/N
        ent+=(-1.0*P*np.log2(P))
    return ent

# a function to divide the data into left and right systems
def divide_data(x_data,fkey,fval):
    #make two empty dataframes for left and right
    x_right=pd.DataFrame([],columns=x_data.columns)
    x_left=pd.DataFrame([],columns=x_data.columns)
    #traverse the data and segregate into left and right depending upoin fvalue/threshold value
    for ix in range(x_data.shape[0]):
        val=x_data[fkey].loc[ix]
        if val>fval:
            x_right=x_right.append(x_data.loc[ix])
        else :
            x_left=x_left.append(x_data.loc[ix])
    return x_left,x_right 

#a function to calculate information gain
def information_gain(x_data,fkey,fval):
    #divide the data
    left,right=divide_data(x_data,fkey,fval)
    l=float(left.shape[0])/x_data.shape[0]
    r=float(right.shape[0])/x_data.shape[0]
    #no further splitting is possible so return minm information gain 
    if left.shape[0]==x_data.shape[0] or right.shape[0]==x_data.shape[0]:
        return -100000000
    i_gain=entropy(x_data[out_cols])-(l*entropy(left.Outcome)+r*entropy(right.Outcome))
    return i_gain

#decision tree Class
class DecisionTree:
    #constructor
    def __init__(self,depth=0,max_depth=5):
        self.left=None
        self.right=None
        self.fkey=None
        self.fval=None
        self.depth=depth
        self.max_depth=max_depth
        self.target=None

    #a function to train the model!
    def train(self,X_train):
        features=input_cols
        #store infor gains of all features and choose the max
        info_gains=[]
        for ix in features:
            i_gain=information_gain(X_train,ix,X_train[ix].mean())
            info_gains.append(i_gain)
        self.fkey=features[np.argmax(info_gains)]
        self.fval=X_train[self.fkey].mean()
        #now split data
        data_left,data_right=divide_data(X_train,self.fkey,self.fval)
        data_left=data_left.reset_index(drop=True)
        data_right=data_right.reset_index(drop=True)
        #stop at a leaf node
        if data_left.shape[0]==0 or data_right.shape[0]==0:
            if X_train.Outcome.mean()>=0.5 :
                self.target="Not Diabetic"
            else :
                self.target="Diabetic"
            return 
        #stop early if max depth reached
        if self.depth>=self.max_depth :
            if X_train.Outcome.mean()>=0.5 :
                self.target="Not Diabetic"
            else :
                self.target="Diabetic"
            return 
        #building tree recursively
        self.left=DecisionTree(depth=self.depth+1, max_depth=self.max_depth)
        self.left.train(data_left)
        self.right=DecisionTree(depth=self.depth+1, max_depth=self.max_depth)
        self.right.train(data_right)
        #set target at every node( to make predictions at mid nodes)
        if X_train.Outcome.mean()>=0.5 :
                self.target="Not Diabetic"
        else :
            self.target="Diabetic"
        return    

    def predict(self,test):
        if(test[self.fkey]>self.fval):
            if self.right is None:
                return self.target
            return self.right.predict(test)
        else :
            if self.left is None:
                return self.target
            return self.left.predict(test)

#initialise decision tree object        
dt=DecisionTree()
#train the model
dt.train(df)

#read the test data
test_data=pd.read_csv('Diabetes_Xtest.csv')

#y_pred stores the predictions made on test data
y_pred=[]
for ix in range(test_data.shape[0]):
    y_pred.append(dt.predict(test_data.loc[ix]))
#store the predictions in a file
with open('submit.csv','w') as f:
    f.write('Id,survived\n')
    for i in range(len(y_pred)):
        f.write("{},{}\n".format(i,y_pred[i]))
