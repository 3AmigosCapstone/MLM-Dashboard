#This script allows the user to build and test a Decision Tree classification model

#Import Libraries
import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeClassifier,plot_tree
import time
import matplotlib.pyplot as plt

#Import custom data cleaning function from data_cleaner.py script
from data_cleaner import *

#Decision Tree Main Function Definition
def DTC_Main(dataset):
    #Page Title
    st.header("Decision Tree Classification")
    st.write("---")
    #Allow the user to build their model
    st.header("Build Your Model")
    #Allow user to select target,features, and split data
    X_train, X_test, y_train, y_test = Data_Cleaning_Main(dataset,'Classification')
    st.write("___")
    #Allow user to train model
    st.header("Train Your Model")
    #Create 2 columns for aesthetics
    col1, col2 = st.columns([3,1])
    #First Column
    with col1:
        #Allow User to select the maximum depth of the decision tree, defaults to the number of features
        depth_input = st.slider("Set a Maximum Depth:",min_value = 1,value = len(list(X_train.columns)), step = 1)
    #Second Column
    with col2:
        #Spacers
        for i in range(2):
            st.write("")
        #Once the user presses this button, the model will be trained and tested
        train_button = st.button("Train Model")
    #Train and test the model
    if train_button:
        #get start time
        startTime = time.time()
        #Fit Decision Tree
        dtree = DecisionTreeClassifier(max_depth = depth_input)
        dtree = dtree.fit(X_train, y_train)
        #Create 3 columns for aesthetics
        col3, col4, col5 = st.columns([5,2,2])
        #First Column
        with col3:
            #Display a graphic of the decision tree
            plt.figure(figsize=(13,13))
            plot_tree(dtree, fontsize = 8, feature_names=X_train.columns)
            st.pyplot(plt)
            #Second Column
        with col4:
            #Display a dataframe of the feature importance of each important feature
            featureImportance = pd.DataFrame(columns=['Feature','Importance'])
            for i,v in enumerate(dtree.feature_importances_):
                if v > 0:
                    featureImportance = pd.concat([featureImportance,pd.DataFrame({'Feature':[list(X_train.columns)[i]],'Importance':[v]})],ignore_index=True)
            featureImportance = featureImportance.sort_values(by='Importance',ascending=False).reset_index(drop=True)
            st.write(featureImportance)
        with col5:
            #Display the training and testing accuracy, as well as how long it took the model to run
            st.write("Training Accuracy: " + str(np.round(100*dtree.score(X_train, y_train),2)) + "%")
            st.write("Testing Accuracy: " +str(np.round(100*dtree.score(X_test, y_test),2)) + "%")
            endTime = time.time()
            st.write("Time to Run: " + str(np.round(endTime-startTime,4)) + " Seconds")
    st.write("___")
