# Stochastic Optimisation Method
# Dom Dent & Karan Mukhi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import sklearn.metrics
import copy
import os
import sys

def get_script_path():
	# return working directory path
    return os.path.dirname(os.path.realpath("StochOpt.py"))

def read(path2):
	# fn. to read in data from given path
    readdata = pd.read_csv(path2, sep=",", header=None);
    data = np.array(readdata);
    data = np.swapaxes(data,0,1);
    #Need to seperate the classification dimension:
    classIndex = np.shape(data)[0] - 1
    classification = data[classIndex]
    labels = np.array(list(set(classification)))
    data = np.delete(data, classIndex, axis=0)
    data = data.astype(float);
    #Normalise data
    for i in range(0, np.shape(data)[0]):
        data[i, :] = data[i, :] - np.min(data[i, :])
        data[i, :] = (data[i, :] / np.ndarray.max(data[i, :])) * 2 - 1
    return data, classification, labels

def writeData(name, tData, tBeta, tHinge):
    n,o = np.shape(data)
    T = len(tData)
    writeData = np.zeros((T,pValue*o))
    writeBeta = np.zeros((T,pValue*n))
    for i in range(pValue):      
        writeBeta[:,i*n:(i+1)*n] = tBeta[:,i,:]
        writeData[:,i*o:(i+1)*o] = tData[:,i,:]
    np.savetxt("/Output/" + name + "tData.txt", writeData, delimiter=",")
    np.savetxt("/Output/" + name+"tBeta.txt", writeBeta, delimiter=",")
    np.savetxt("/Output/" + name+"tCost.txt", tHinge, delimiter=",")

def iterator(X, G, tau):
    """
    Computes Y(tau) or X^{t+1} given X^t and G
    """
    
    if np.shape(X)[1] > 0.5 * np.shape(X)[0]:
        I = np.identity(np.shape(X)[0])
        W = np.matmul(G, X.T) - np.matmul(X, G.T)
        term = (I + (tau / 2) * W)
        Y = np.matmul(np.linalg.inv(term), np.matmul(term, X))
    else:
        I = np.identity(np.shape(X)[1]*2)
        U = np.concatenate((G, X), axis=1)
        V = np.concatenate((X, -G), axis=1)
        B = np.identity(np.shape(X)[1]*2) + (tau / 2) * np.matmul(V.T, U)
        B = np.linalg.inv(B)
        B = np.matmul(U,B)
        A = np.matmul(V.T,X)
        Y = X - tau*np.matmul(B,A)
    return Y

def derivative(X, costfn, eps = 0.0001):
    G = np.zeros((np.shape(X)[0], np.shape(X)[1]))
    for i in range(np.shape(X)[0]):
        for j in range(np.shape(X)[1]):
            A = copy.deepcopy(X)
            A[i,j] += eps
            G[i,j] = (costfn(A) - costfn(X)) / eps
    return G

def hingeLoss(nData):
    clf = OneVsRestClassifier(SVC(kernel='linear'), n_jobs=-1)
    clf.fit(nData, classification)
    prob = clf.decision_function(nData)
    score = sklearn.metrics.hinge_loss(classification,prob,labels)
    return score

def costfn(X):
    xData = transform(X,data)
    w = hingeLoss(xData)
    return w

def transform(X, data):
    xData = np.dot(data.T,X)
    return xData

def SVM(X):
    nData = transform(X,data)
    clf = OneVsRestClassifier(SVC(kernel='linear'), n_jobs=-1)
    clf.fit(nData, classification)
    score = clf.score(nData, classification)
    return score

def sigma(sig,i):
    s = sig
    return s;

def noise(X,var3):
    beta = 1 - np.sqrt(2)/2
    I = np.identity(np.shape(X)[0])
    A = beta*np.matmul(X,X.T)
    A = I-A
    B = np.random.normal(0,var3,(np.shape(X)))
    return np.matmul(A,B)

def optimise(data, metric, p, T, tau, sig, var2):
    '''
    costfunc - the function to be minimised
    p - number of dimensions to project onto
    T - number of steps
    tau - step size on Steifel manifold
    sig - the strength of the noise,
    either constant numerical number or function
    var - varience of the gaussian noise
    '''
    minHinge = 10
    maxAcc = 0
    n = np.shape(data)[0] #native dimensions of data
    o = np.shape(data)[1] #number of data points
    X = np.zeros((n,p))  #transformation matrix
    xData = np.zeros((o,p)) 
    tX = np.zeros((T,n,p)) #array storing trasformation matrix at each timestep
    tMinX = np.zeros((0,n,p))
    tMinX0 = np.zeros((1,n,p))
    tData = np.zeros((T,o,p)) #array storing transformed data at each timestep
    tHinge = np.zeros(T) #HingeLoss function at each timestep
    jump = [] #array storing timesteps of minima
    start = time.time() 
    for i in range (p):
        X[i,i] = 1
    for i in range(T):
        G = derivative(X, costfn) #calculate derivative array
        Z = G + sigma(sig, i) * noise(X, var2) #step direction array from derivative including noise
        X = iterator(X,Z,tau) #form new transformation matrix
        w = costfn(X) #calculate hingeloss at new step
        tX[i,:,:] = X #store transformation matrix
        tData[i,:,:] = transform(X,data) #store transformed data
        tHinge[i] = w #store hingeloss
        if  w < minHinge: #if new optimal position found 
            minHinge = w #change new minimum
            minX = X #store minimum transformation matrix
            tMinX0[0] = X 
            tMinX = np.append(tMinX,tMinX0,axis = 0) #store in a matrix 
            maxAcc = SVM(X) #calculate minimum accuracy
            jump.append(i) #store timestep of minima
        end = time.time()
        if i == 0:
            continue
        remaining_time = ((end - start) / i) * (1000 - i)
        remaining_time = round(remaining_time / 60, 2)
        time_elapsed = round((end - start), 2)
        end_time = ((remaining_time * 60 + time_elapsed) * (total_loops - loop_counter)) / 60
        end_time = round(end_time, 2)
        print("Iteration: " + str(i) + "/1000, time elapsed: " + str(time_elapsed) + " seconds. Remaining (loop): " + str(remaining_time) + " mins. End in: " + str(end_time) + " mins", end = "\r")
        
        # print("Time:", (end-start)*(T-i), costfn(X), SVM(X),", Max (Hinge, Accuracy): ", minHinge, maxAcc, end = "\r")
    tData = np.swapaxes(tData,1,2) #transpose tData for plotting
    tX = np.swapaxes(tX,1,2) #transpose tX for plotting
    print("Done.")
    return tData, tX, tHinge

# Global variable for path
path = get_script_path()

# input the parameters you want to test for 
datasets = ["wine"]
tauValues = [0.1, 0.5, 1, 1.5]
sigmaValues = [1]
varianceValues = [1]

pValue = 2
timesteps = 1000 

num_loops = 10

total_loops = len(tauValues) * len(datasets) * len(sigmaValues) * len(varianceValues) * num_loops

loop_counter = 1

# loop through all values for parameters chosen
for current_dataset in datasets:
    print("Number of loops to calculate: " + str(total_loops))
    print("Current dataset: " + str(current_dataset))
    for current_tau in tauValues:
        print("Current tau value: " + str(current_tau))
        for current_sigma in sigmaValues:
            print("Current sigma value: " + str(current_sigma))
            for current_var in varianceValues:
                print("Current variance value: " + str(current_var))
                for loop_num in range(num_loops):

                    print("Loop number: " + str(loop_counter) + "/" + str(total_loops))
                    name = current_dataset + "_" + str(current_tau) + "_" + str(current_sigma) + "_" + str(current_var) + "_" + str(pValue) + "_" + str(timesteps) + "_l" + str(loop_num)
                    data, classification, labels = read(path + "/Data/" + current_dataset + "Data.txt")
                    tData, tBeta, tHinge = optimise(data, hingeLoss, pValue, timesteps, current_tau, current_sigma, current_var)
                    writeData(name, tData, tBeta, tHinge)
                    loop_counter += 1
