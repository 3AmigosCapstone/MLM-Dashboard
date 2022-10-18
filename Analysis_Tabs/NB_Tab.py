#This script allows the user to build and test a Naive Bayes classification model

#Import Libraries
import streamlit as st
from sklearn.naive_bayes import GaussianNB
import time
import numpy as np

#Import custom data cleaning function from data_cleaner.py script
from data_cleaner import *

#Naive Bayes Main Function Definition
def NB_Main(dataset):
    #Page Title
    #Allow the user to build their model
    st.header("Build & Train Your Model")
    #Allow user to select target,features, and split data
    st.write("---")
    X_train, X_test, y_train, y_test = Data_Cleaning_Main(dataset,'Classification')
    #Once the user presses the button, the model will be trained and tested
    train_button = st.button("Train Model")
    #Train and test the model
    if train_button:
        #get start time
        startTime = time.time()
        #Fit Gaussian Naive Bayes Classifier
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        #Display testing and training accuracy
        st.subheader("Training Accuracy:" + str(np.round(100*gnb.score(X_train, y_train),2)) + "%") #Accuracy of the model when training.
        st.subheader("Testing Accuracy:" + str(np.round(100*gnb.score(X_test, y_test),2)) + "%" )
        #Display how long it took to run
        endTime = time.time()
        st.write("Time to Run: " + str(np.round(endTime-startTime,4)) + " Seconds")
    st.write("___")
