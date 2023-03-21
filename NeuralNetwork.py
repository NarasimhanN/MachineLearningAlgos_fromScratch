STUDENT_NAME = "Narasimhan_N" #Put your name
STUDENT_ROLLNO = "MT2022062" #Put your roll number
CODE_COMPLETE = True 
# set the above to True if you were able to complete the code
# and that you feel your model can generate a good result
# otherwise keep it as False
# Don't lie about this. This is so that we don't waste time with
# the autograder and just perform a manual check
# If the flag above is True and your code crashes, that's
# an instant deduction of 2 points on the assignment.
#
#@PROTECTED_1_BEGIN
## No code within "PROTECTED" can be modified.
## We expect this part to be VERBATIM.
## IMPORTS 
## No other library imports other than the below are allowed.
## No, not even Scipy
import numpy as np 
import pandas as pd 
import sklearn.model_selection as model_selection 
import sklearn.preprocessing as preprocessing 
import sklearn.metrics as metrics 
from tqdm import tqdm # You can make lovely progress bars using this

## FILE READING: 
## You are not permitted to read any files other than the ones given below.
X_train = pd.read_csv("train_X.csv",index_col=0).to_numpy()
y_train = pd.read_csv("train_y.csv",index_col=0).to_numpy().reshape(-1,)
X_test = pd.read_csv("test_X.csv",index_col=0).to_numpy()
submissions_df = pd.read_csv("sample_submission.csv",index_col=0)
#@PROTECTED_1_END

X = X_train
y = y_train

y = pd.get_dummies(y)
y=np.array(y)


#Scaling not performed as Accuracy was better without Scaling 
'''Types of Scaling Performed before submission : 
1)Standard Scaling
2)Minmax Scaling
3) X=X/255

'''
X_train, X_tt, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=0.4)

X_train = X_train.T
X_tt = X_tt.T
Y_train = Y_train.T
Y_test = Y_test.T
 
class NN:
    
    def __init__(self,X, noHiddenNode, Y,learning_rate,iterations):
        '''To initilise all the values used by model'''
        self.training_data = X
        self.trainig_result = Y
        noInputNode = X.shape[0]
        noOutputNode = Y.shape[0]
        #0.01 for tan and 0.001 for relu
        w1 = np.random.randn(noHiddenNode, noInputNode)*0.001
        b1 = np.zeros((noHiddenNode, 1))

        w2 = np.random.randn(noOutputNode, noHiddenNode)*0.001
        b2 = np.zeros((noOutputNode, 1))

        self.parameters = {
            "w1" : w1,
            "b1" : b1,
            "w2" : w2,
            "b2" : b2
        }
        self.forward_pass_paramaters = {
            "z1" : 0,
            "a1" : 0,
            "z2" : 0,
            "a2" : 0
        }
        
        self.gradients = {
            "dw1" : 0,
            "db1" : 0,
            "dw2" : 0,
            "db2" : 0
        }
        
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.cost_list = list()

    def relu(self,x):
        '''Activation Function : Relu max(x,0)'''
        return np.maximum(x, 0)

    def softmax(self,x):
        '''Activation Function at the Output layer'''
        expX = np.exp(x)
        return expX/np.sum(expX, axis = 0)

    def derivative_relu(self,x):
        '''Derivative of Relu, max(1,0)'''
        return np.array(x > 0, dtype = np.float32)

    def cost_OutputLayer(self):
        '''Cost at the output layer'''
        m = self.trainig_result.shape[1]
        a2 = self.forward_pass_paramaters['a2']
        
        cost = -(1/m)*np.sum(self.trainig_result*np.log(a2))

        #cost = -(1/m)*np.sum(np.sum(self.trainig_result*np.log(a2, 0), 1))
        self.cost_list.append(cost)
        return cost

    def update_parameters(self):
        '''For updating the paramaters during back propagation'''
        w1 = self.parameters['w1']
        b1 = self.parameters['b1']
        w2 = self.parameters['w2']
        b2 = self.parameters['b2']

        dw1 = self.gradients['dw1']
        db1 = self.gradients['db1']
        dw2 = self.gradients['dw2']
        db2 = self.gradients['db2']

        lr = self.learning_rate
        
        w1 = w1 - lr*dw1
        b1 = b1 - lr*db1
        w2 = w2 - lr*dw2
        b2 = b2 - lr*db2
    
        self.parameters['w1']=w1
        self.parameters['b1']=b1
        self.parameters['w2']=w2
        self.parameters['b2']=b2

    def backward_prop(self):
    
        w1 = self.parameters['w1']
        b1 = self.parameters['b1']
        w2 = self.parameters['w2']
        b2 = self.parameters['b2']

        a1 = self.forward_pass_paramaters['a1']
        a2 = self.forward_pass_paramaters['a2']

        m = self.training_data.shape[1]
    
        y = self.trainig_result
        x= self.training_data
        
        # Derivative on Output Cost :
        dz2 = (a2 - y)
        dw2 = (1/m)*np.dot(dz2, a1.T)
        db2 = (1/m)*np.sum(dz2, axis = 1, keepdims = True)

        dz1 = (1/m)*np.dot(w2.T, dz2)*self.derivative_relu(a1)
        dw1 = (1/m)*np.dot(dz1, x.T)
        db1 = (1/m)*np.sum(dz1, axis = 1, keepdims = True)

        self.gradients['dw1']=dw1
        self.gradients['db1']=db1
        self.gradients['dw2']=dw2
        self.gradients['db2']=db2


    def forward_propagation(self):
        '''Forward Propagation'''
        w1 = self.parameters['w1']
        b1 = self.parameters['b1']
        w2 = self.parameters['w2']
        b2 = self.parameters['b2']

        z1 = np.dot(w1, self.training_data) + b1
        a1 = self.relu(z1)

        z2 = np.dot(w2, a1) + b2
        a2 = self.softmax(z2)

        self.forward_pass_paramaters['z1']=z1
        self.forward_pass_paramaters['a1']=a1
        self.forward_pass_paramaters['z2']=z2
        self.forward_pass_paramaters['a2']=a2
        
    def get_output_activation(self,input_data):
        '''Returns output layer activation'''
        w1 = self.parameters['w1']
        b1 = self.parameters['b1']
        w2 = self.parameters['w2']
        b2 = self.parameters['b2']

        z1 = np.dot(w1, input_data) + b1
        a1 = self.relu(z1)

        z2 = np.dot(w2, a1) + b2
        a2 = self.softmax(z2)
        return a2
        
    def get_predicted_labels(self,input_data):
        '''Returns the predicted labels given input data'''
        output_activation = self.get_output_activation(input_data)
        predicted_labels = np.argmax(output_activation, 0)
        return predicted_labels
    
    def get_accuracy(self,input_data, actual_labels):
        '''Returns the accuracy between predicted values and actual_labels'''
        
        predicted_labels = self.get_predicted_labels(input_data)
        actual_labels = np.argmax(actual_labels, 0)
        accuracy = np.mean(predicted_labels== actual_labels)*100
        return accuracy
        
    def train_model(self):
        lr =self.learning_rate
        for i in range(self.iterations):
            if(i<1):
                print('Initial iteratonnn')
                self.learning_rate = 5
            else:
                self.learning_rate = lr
            self.forward_propagation()
            cost = self.cost_OutputLayer()
            self.backward_prop()
            self.update_parameters()
            
            if(i%100 == 0):
                print("Cost after", i, "iterations is :", cost)

                
no_HiddenLayerNodes = 1024

nn = NN(X_train,no_HiddenLayerNodes,Y_train,learning_rate=0.2,iterations=2000)
nn.train_model()
print("------------------------------")
print("Train Accuracy : ",nn.get_accuracy(X_train,Y_train))

print("Test Accuracy : ",nn.get_accuracy(X_tt,Y_test))

#Checking for given data : 
X_test = X_test.T
predicted_labels = nn.get_predicted_labels(X_test)
submissions_df['label'] = predicted_labels

#@PROTECTED_2_BEGIN 
##FILE WRITING:
# You are not permitted to write to any file other than the one given below.
submissions_df.to_csv("{}__{}.csv".format(STUDENT_ROLLNO,STUDENT_NAME))
#@PROTECTED_2_END