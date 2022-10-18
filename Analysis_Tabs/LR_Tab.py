#This script allows the user to build and test a Logistic Regression model

#Import Libraries
import streamlit as st
from sklearn.linear_model import LogisticRegression
import time
import numpy as np

#Import custom data cleaning function from data_cleaner.py script
from data_cleaner import *

#Logistic Regression Main Function Definition
def LR_Main(dataset):
    #Page Title
    st.header("Linear Regression")
    st.write("---")
    #Allow the user to build their model
    st.header("Build Your Model")
    #Allow user to select target,features, and split data
    X_train, X_test, y_train, y_test = Data_Cleaning_Main(dataset,'Regression')
    st.write("___")
    #Once the user presses the button, the model will be trained and tested
    train_button = st.button("Train Model")
    #Train and test the model
    if train_button:
        #get start time
        startTime = time.time()
        #Fit Logistic Regression model
        logreg = LogisticRegression()
        logreg = logreg.fit(X_train, y_train)
        #Display testing and training accuracy
        st.subheader("Training Accuracy: " + str(np.round(100*logreg.score(X_train, y_train),2)) + "%") #Accuracy of the model when training.
        st.subheader("Testing Accuracy: " + str(np.round(100*logreg.score(X_test, y_test),2)) + "%" )
        #Display how long it took to run
        endTime = time.time()
        st.write("Time to Run: " + str(np.round(endTime-startTime,4)) + " Seconds")
