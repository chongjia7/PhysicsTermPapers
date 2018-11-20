#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:13:22 2017

@author: chongjiayoong
"""
########################################################################################
#KAGGLE MACHINE LEARNING CHALLENGE: DROP-OUT NEURAL NETWORK FOR HIGGS BOSON DETECTION  #
########################################################################################

#import useful modules from numpy and math, and matplotlib 
#import tensorflow for implementation of neural network 
#import sci-kit learn 
#import cross_validation in order for cross validation purpose 
import numpy as np
import math
import os, sys
from matplotlib import pyplot as plt
import tensorflow.python.platform
import tensorflow as tf
import sklearn
from sklearn.cross_validation import StratifiedKFold
from sklearn import preprocessing

#define useful constants for learning rate, and iteration number, and number of neuron nodesand of output 
learning_rate = 0.02 
N = 10000 
n_neurons = 600 
n_output = 2 
# Stratified 5-fold Shuffling for Cross Validation:
cross_validation = sklearn.cross_validation.StratifiedKFold(Y_data, n_folds=5, shuffle=True, random_state=None)
# Load traning data:
training = np.loadtxt('training.csv', delimiter=',', skiprows= 1, converters={32: lambda x:int(x=='s'.encode('utf-8'))})
test = np.loadtxt('test.csv', delimiter=',', skiprows=1)

#Use one tenth of the original data set for training and test set 
training_set = training[0:int(len(training)/10)] 
test_set = test[0:int(len(test)/10)]

#delete the input phi features 
delete_list = [16,18,19,23,25]
for i in delete_list:
    training_set = np.delete(data_train,i,1)
    test_set = np.delete(data_test,i,1)

#test set for input features 
X_test = test_set[:,1:test_set.shape[1] - 1]
# classifying data into Y(labels), X(input), W(weights)
Y_data = training_set[:,training_set.shape[1]-1] > 0.
#obtain all the features from 1 to max_idx-2 
X_data = training_set[:,1:training_set.shape[1]-3]
#weight is on the last column 
W_data = training_set[:,training_set.shape[1]-2]

#Normalize Dataset
#The mean of the data set is made to be 0
#The standard deviation of the data set is made to be 1 
#define a function to obtain the mean and standard deviation of the array 
X_data = preprocessing.scale(X_data)
X_test = preprocessing.scale(X_test)

#define the droput neural network and set it to 
# 26 x 600 x 600 x 600 x 2 with three hidden layers. 
def dropout_neural_network(n_neurons,n_output):
    
        def init_weights(dimension):
            return tf.Variable(tf.random_normal(dimension, stddev=0.01))
    
        X = tf.nn.dropout(tf.placeholder("float", [None, n_neurons]), tf.placeholder("float"))
        h1 = tf.nn.relu(tf.matmul(X, init_weights([X_data.shape[1], n_neurons])))
        
        h1 = tf.nn.dropout(h1, tf.placeholder("float"))
        h2 = tf.nn.relu(tf.matmul(h1, init_weights([n_neurons, n_neurons]) ))
        
        h2 = tf.nn.dropout(h2, tf.placeholder("float"))
        h3 = tf.nn.relu(tf.matmul(h2, init_weights([n_neurons, n_neurons]) ))
        
        h3 = tf.nn.dropout(h3, tf.placeholder("float"))
        
        return tf.matmul(h3,init_weights([n_neurons, n_output]))
    
dnn = dropout_neural_network(600,2)
#define the cost function by using reduce_mean and softmax_cross_entropy_with_logits
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = dnn,labels = tf.placeholder("float", [None, 2])))
#define training optimizer by using RMSPropOptimizer.minimize with 
#parameter of learning rate set 0.02 
train_optimizer = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(cost)
#define predicting optimizer by softmax function on the dropout neural network 
p_optimizer = tf.nn.softmax(dropout_neural_network) 

#running the tensorflow computational graphs session as initilized 
#initialize global variables 
with tf.Session() as sess: 
    
    sess.run(tf.initialize_all_variables())
    
    #define the AMS score for training set and validation set 
    def AMS_train(prob_predict_train, W_train, ratio):
        pcut = np.percentile(prob_predict_train,80)
        TF_array = (prob_predict_train > pcut) == 1.0
        s_train = sum ( W_train * (Y_train == 1.0) * (1.0/ratio) * ((TF_array)))
        b_train = sum ( W_train * (Y_train == 0.0) * (1.0/ratio) * ((TF_array)))
        return math.sqrt (2.*( (s_train + b_train + 10.) * math.log(1. + s_train / (b_train + 10.)) - s_train))
    
    def AMS_valid(prob_predict_valid, W_valid, ratio):
        pcut = np.percentile(prob_predict_train, 80)
        s_valid = sum ( W_valid * (Y_valid == 1.0) * (1.0/(1-ratio)) * ((prob_predict_valid > pcut) == 1.0))
        b_valid = sum ( W_valid * (Y_valid == 0.0) * (1.0/(1-ratio)) * ((prob_predict_valid > pcut) == 1.0))
        return math.sqrt (2.*( (s_valid + b_valid + 10.) * math.log(1. + s_valid / (b_valid + 10.)) - s_valid))
   
    #define a function that convert object in array to either True or empty(False)
    def binarising(array):
        binary = np.zeros(shape=(len(array), 2))
        for i in range(len(array)):
            if (array[i]):
                binary[i][1] = True
            else:
                binary[i][0] = True 
        return binary
         
    sess.run(tf.initialize_all_variables())
    
    #training the model by shuffling the validation and training set
    #print the AMS score for each iteration until N
    for i in range(N):
        for t, v in cross_validation:
            
            sess.run(tf.initialize_all_variables())
            
            X_train = X_data[t]
            Y_train = Y_data[t]
            W_train = W_data[t]
            
            X_valid = X_data[v]
            Y_valid = Y_data[v]
            W_valid = W_data[v]
            
            input_holder = tf.placeholder("float")
            hidden_holder = tf.placeholder("float")
            
            sess.run(train_optimizer, feed_dict={X: X_train, Y: binarising(Y_train), input_holder: 0.8, hidden_holder: 0.5})
           
		   #run tensorflow session in order to obtain the probability predicted from validation set and training set 
            prob_train = sess.run(p_optimizer, feed_dict={X: X_train, Y: binarising(Y_train), input_holder: 1.0, hidden_holder: 1.0})[:,1]
            
            prob_valid = sess.run(p_optimizer, feed_dict={X: X_valid, Y: binarising(Y_valid), input_holder: 1.0, hidden_holder : 1.0})[:,1]
            
            ratio = len(X_train)/(len(X_train) + len(X_valid))
            
            ams_train = AMS_train(prob__train, W_train, ratio)
            
            ams_valid = AMS_valid(prob_valid, W_valid, ratio)
            print("Iteration number:", i)
            print("AMS score for training set is", ams_train)
            print("AMS score for validation set is", ams_valid)

    #train the classifier by training the optimizer through feeding the data
    with tf.Session as sess:
        
        sess.run(tf.initialize_all_variables())
        
        signal_classifier = sess.run(p_optimizer, feed_dict={X: X_train[Y_train>0.5], input_holder: 1.0, hidden_holder: 1.0})[:,1].ravel()
        
        background_classifier = sess.run(p_optimizer, feed_dict={X: X_train[Y_train<0.5], input_holder: 1.0, hidden_holder: 1.0})[:,1].ravel()
        
        classifier_testing_a = sess.run(p_optimizer, feed_dict={X: X_test, input_holder: 1.0, hidden_holder: 1.0})[:,1].ravel()
        
        histo_training_s = np.histogram(signal_classifier, bins=50, range=(0.49,0.51))
        
        histo_training_b = np.histogram(background_classifier, bins=50, range=(0.49,0.51))
        
        #plot the signal and background distribution 
        plt.hist(histo_training_s[0]/max(histo_training_s[0]),color ="deeppink")
        
        plt.hist(histo_training_b[0]/max(histo_training_b[0]), color ="darkorchid")

