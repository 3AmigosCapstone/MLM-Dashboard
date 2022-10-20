#This script allows the user to build and test a k-nearest neighbors classification model

#Import Libraries
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

from data_cleaner import *


#k-Nearest Neighbor Main Function Definition
def RandomForest_Main(dataset):
    #Page Title
    st.header("Random Forest Classification")
    st.write("---")

    #Allow the user to build their model
    st.header("Build Your Model")
    #Allow user to select target,features, and split data
    X_train, X_test, y_train, y_test = Data_Cleaning_Main(dataset,'Classification')
    st.write("___")

    #Allow user to train model
    st.header("Train Your Model")
    #Creates 2 columns for aesthetics
    col1, col2 = st.columns([3,1])
    #First Column
    with col1:
        #Allow User to Select k value, defaults to 1
        jobs = st.number_input("Choose number of jobs:",min_value = 1, value = 2, step = 1)
        estimators = st.number_input('Choose the number of estimators:', min_value=2, value=5, step=1)
    #Second Column
    with col2:
        #Spacers
        for i in range(2):
            st.write("")
        #Once the user clicks this button it will train and test the model they've built
        train_button = st.button("Train Model")
    #Train and test the model the user has built
    if train_button:
        #get start time
        startTime = time.time()
        #Declare and fit model using the k and the data that the user selected
        forest = RandomForestClassifier(criterion='gini',
                                 n_estimators=estimators,
                                 n_jobs=jobs)
        forest=forest.fit(X_train, y_train)
        #Display the testing and training accuracy to the user
        st.write("Training Accuracy: " + str(np.round(100*forest.score(X_train, y_train),2)) + "%")
        st.write("Testing Accuracy: " +str(np.round(100*forest.score(X_test, y_test),2)) + "%")
        #Get the end time and display to the user how long it took for the model to fit and test
        endTime = time.time()
        st.write("Time to Run: " + str(np.round(endTime-startTime,4)) + " Seconds")
    st.write("___")
    #Allow user to grid search their linear_model
    st.header("Grid Search Your Model")
    st.write("*Note: Grid Search on Random Forest can take a considerable amount of time depending on the number of parameters to test.*")
    #Create 4 columns for aesthetics
    col5, col6, col7, col8,col9 = st.columns([2,2,2,2,1])
    #First Column
    with col5:
        #Allow user to select max features
        max_feature_list = []
        st.write("Select Max Features:")
        for x in ['auto','sqrt','log2']:
            if st.checkbox(x,value=True):
                max_feature_list.append(x)
    #Second Column
    with col6:
#        Allow user to select max features
        criterion_list = []
        st.write("Select Criterion:")
        for x in ['gini','entropy']:
            if st.checkbox(x,value=True):
                criterion_list.append(x)
    #Third Column
    with col7:
        #Allow user to select c values. Let them choose a minimum, maximum, and the size of the interval in between steps
        c_interval = st.number_input("Select Estimator Number Interval Size:", min_value = 1, value = 1)
        c_min = st.number_input('Minimum Number of Estimators: ',min_value=1,value=1,step = 1)
        c_max = st.number_input('Maximim Number of Estimators: ',min_value=int(c_min),value=int(c_min + c_interval),step = int(c_interval))
        #If the max and min are at least one interval apart, create a list of c values that contains as many intervals as possible between the max and min
        if c_max-c_min >= c_interval:
            c_values = []
            c_i = c_min
            while c_i <= c_max:
                c_values.append(c_i)
                c_i += c_interval
    #Third Column
    with col8:
        #Allow user to select gamma values. Let them choose a minimum, maximum, and the size of the interval in between steps
        gamma_interval = st.number_input("Select Max Depth Interval Size:",min_value = 1, value = 1)
        gamma_min = st.number_input('Minimum Gamma: ',min_value=1,value=1,step = 1)
        gamma_max = st.number_input('Maximum Gamma: ',min_value=int(gamma_min),value=int(gamma_min + gamma_interval),step = int(gamma_interval))
        #If the max and min are at least one interval apart, create a list of c values that contains as many intervals as possible between the max and min
        if gamma_max-gamma_min >= gamma_interval:
            gamma_values = []
            gamma_i = gamma_min
            while gamma_i <= gamma_max:
                gamma_values.append(gamma_i)
                gamma_i += gamma_interval
    #Fourth Column
    with col9:
        #Make sure the data works by ensuring that there is at least one kernel, and that the intervals, maximums, and minimums work together
        if len(max_feature_list) == 0:
            st.warning("Must select at least one max feature")
        elif len(criterion_list) == 0:
            st.warning("Must select at least one criterion")
        elif c_max-c_min < c_interval:
            st.warning("Estimator Number Minimum and Maxmimum must contain at least one interval")
        elif gamma_max-gamma_min < gamma_interval:
            st.warning("Max Depth Minimum and Maxmimum must contain at least one interval")
        #If the selected options are all valid, create a button that will run the grid search when the user clicks it
        else:
            grid_search_button = st.button("Run Grid Search")
    #Run the Grid search
    if grid_search_button:
        #get start time
        startTime = time.time()
        #Create a parameter grid based on the users selections
        param_grid = {'max_features' : max_feature_list, 'criterion':criterion_list,'n_estimators':c_values,'max_depth':gamma_values}
        #Perform the grid search
        grid_search = GridSearchCV(RandomForestClassifier(), param_grid, return_train_score = True)
        grid_search.fit(X_train, y_train)
        #Get the best parameters for the grid search
        best_params = grid_search.best_params_
        #Display the best parameters to the user
        st.subheader("Best Max Features: " + best_params['max_features'])
        st.subheader("Best Criterion: " + str(best_params['criterion']))
        st.subheader("Best Number of Estimators Value: " + str(best_params['n_estimators']))
        st.subheader("Best Max Depth: " + str(best_params['max_depth']))
        #Display the Training and testing accuracy of the best model to the user
        st.subheader("Training Accuracy:" + str(np.round(100*grid_search.score(X_train, y_train),2)) + "%")
        st.subheader("Testing Accuracy:" + str(np.round(100*grid_search.score(X_test, y_test),2)) + "%" )
        #Display how long it took to run the grid search
        endTime = time.time()
        st.write("Time to Run Grid Search: " + str(np.round(endTime-startTime,4)) + " Seconds")
