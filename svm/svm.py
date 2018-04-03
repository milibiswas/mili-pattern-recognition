#########################################################################
##
##    Name: svm.py
##    Description : This program implements support vector machine
##                  to classify unkown vectors based on learned machine
##                  algorithm with training feature vectors class.
##
##    Written By  : Mili Biswas (MSc Computer Science)
##    Organization: University of Fribourg
##    Date : 30.03.2018
##
##########################################################################

#==========================
#   Modules Import
#==========================
import numpy as np
import cvxopt as cvx
import math
import operator
import csv
import matplotlib.pyplot as plt

'''

    Algorithm for Mclass SVM
       Input Data : 
        1. n dimensional feature vectors from Training Set
        2. n dimensional vectors as test.
        3. each training data vector will be provided with y (+1 or -1) based on class
       Output:
        1. Linear Discriminant Function parameters
        2. Classification of Test Data Vector
        3. Graphical Representation of Test Data Vector classification
    Step1:
    
      Calculate Kernel for the processing
      
    Step2:
      Using the training data vector and Kernel from Step1 solve the 
      quadratic optimization problem for Langrange Multipliers and subsequently
      calculate the weight vector W for optimal solution and b for bias.
      
      Here use cvxopt solver.
      
    Step3:
    
    ........ To be determined
    

'''
class svm:
        def __init__(self,kernel='linear',C=None,sigma=0.,threshold=0.1):
                '''
                     This is constructor which will get values of different varaibales---
                     ---- kernel    : Type of Kernel
                     ---- C         : Slack Variable
                     ---- sigma     : Standard Deviation
                     ---- threshold : Threshold value to choose lamda (langrange multipliers from solver) where lamdas > threshold
                '''
                self.kernel = kernel
                self.sigma = sigma
                self.C = C
                self.threshold = threshold
                
        def __kernel__(self,l_x):
                '''
                   This function will build the kernel and gives output kernel
                --- Linear Kernel
                --- Polynomial
                --- RBF   
                '''
                
                self.k = np.dot(l_x,l_x.T)
                
        def learn_module(self,l_x,l_target):
                '''
                   This function will train the claissifier based on Training Data.
                   Quadratic solver will be called in this function.
                   
                   §parameters§
                   
                      l_x      : Training Data without class identification
                      l_target : +1 / -1 (based on class pair) 
                '''
                self.n = np.shape(l_x)[0]
                self.__kernel__(l_x)
                P =np.dot(l_target,l_target.T)*self.k
                q = -np.ones((self.n,1))
                G = -np.eye(self.n)
                h = np.zeros((self.n,1))
                b = 0.0
                A = l_target.reshape(1,self.n)
                
                #  Quadratic Solver
                
                sol = cvx.solvers.qp(cvx.matrix(P),cvx.matrix(q),cvx.matrix(G),cvx.matrix(h), cvx.matrix(A), cvx.matrix(b))
                
                
                l_lamda = np.array(sol['x'])   # This holds lamda value
                '''
                find support vector
                '''
                self.sv = np.where(l_lamda>self.threshold)[0]
                
                self.nsupport = len(self.sv)
                print(self.sv)
                print(l_lamda)
        
def read_data():
        '''
        this function is for reading data from input file
        '''
        with open("train.csv",'r') as l_csvfile:
                l_train = csv.reader(l_csvfile)
                l_train_data = np.array(list(l_train),dtype = int)[0:5000,1:]
        return l_train_data
if __name__ == "__main__":
                
        l_data = read_data() 
        l_test_data = [[1,1],[1,2],[2,1],[0,0],[1,0],[0,1]]
        l_test = np.array(l_test_data)
        l_target = -np.ones((6,1))
        l_target[0:3,0]= 1
        obj_svm = svm()
        print(l_data)
        obj_svm.learn_module(l_test,l_target)