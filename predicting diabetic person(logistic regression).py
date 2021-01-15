import pandas as pd
import numpy as np
X_train=pd.read_csv('Diabetes_XTrain.csv')
Y_train=pd.read_csv('Diabetes_YTrain.csv')
X_test=pd.read_csv('Diabetes_XTest.csv')
X_train=np.array(X_train)
Y_train=np.array(Y_train)
X_test=np.array(X_test)
def hypothesis(x,w,b):
    
    h = np.dot(x,w) + b
    return sigmoid(h)

def sigmoid(z):
    
    return 1.0/(1.0 + np.exp(-1.0*z))


def get_grads(y_true,x,w,b):
    
    grad_w = np.zeros(w.shape)
    grad_b = 0.0
    
    m = x.shape[0]
    
    for i in range(m):
        hx = hypothesis(x[i],w,b)
        
        grad_w += (y_true[i] - hx)*x[i]
        grad_b +=  (y_true[i]-hx)
        
    
    grad_w /= m
    grad_b /= m
    
    return [grad_w,grad_b]


# One Iteration of Gradient Descent
def grad_descent(x,y_true,w,b,learning_rate=0.1):
    
    [grad_w,grad_b] = get_grads(y_true,x,w,b)
    
    w = w + learning_rate*grad_w
    b = b + learning_rate*grad_b
    
    return w,b
    
def predict(x,w,b):
    
    confidence = hypothesis(x,w,b)
    if confidence<0.5:
        return 0
    else:
        return 1
    
W = 2*np.random.random((X_train.shape[1],))
b = 5*np.random.random()
for i in range(1000):
    W,b = grad_descent(X_train,Y_train,W,b,learning_rate=0.1)

def predict(x,w,b):
    
    confidence = hypothesis(x,w,b)
    if confidence<0.5:
        return "Diabetic"
    else:
        return "Not Diabetic"

y_pred=[]
for i in range(X_test.shape[0]):
    y_pred.append(predict(X_test[i],W,b))
with open('submit.csv','w') as f:
    f.write('Id,Outcome\n')
    for i in range(len(y_pred)):
        f.write("{},{}\n".format(i,y_pred[i]))