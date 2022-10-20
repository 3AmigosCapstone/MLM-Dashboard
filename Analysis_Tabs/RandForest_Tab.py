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

    # #Allow user to grid search their model for the best k value
    # st.header("Grid Search Your Model")
    # #Allow user to select the starting and ending k values for the grid search
    # grid_start = int(st.number_input('Starting k:',value = 1, min_value = 1,step=1))
    # grid_end = st.number_input('Starting k:',value = int(grid_start + 10), min_value = int(grid_start+1),step=1)
    # #Once the user clicks this button, the grid search will run
    # grid_search_button = st.button("Run Grid Search")
    # #Run the grid search
    # if grid_search_button:
    #     #get start time for grid search
    #     startTime = time.time()
    #     #create the first model in the grid search. Train it and set it as the current best k-value.
    #     best_k = grid_start
    #     knn = KNeighborsClassifier(grid_start)
    #     knn.fit(X_train, y_train)
    #     #Create a list of the training and testing accuracy for the first k value, set the current scores as the best scores
    #     train_score_array = [knn.score(X_train, y_train)]
    #     test_score_array = [knn.score(X_test, y_test)]
    #     best_k_train_score = train_score_array[0]
    #     best_k_test_score = test_score_array[0]
    #     #Create a list of k values that will serve as an x-axis for the plot of accuracies
    #     x_axis = [grid_start]
    #     #iterate through the remaining k values in the grid search
    #     for x in range(grid_start+1,grid_end+1):
    #         #Add k-value to list of k-values
    #         x_axis.append(x)
    #         #Declare and train model
    #         knn = KNeighborsClassifier(x)
    #         knn.fit(X_train, y_train)
    #         #add accuracies to lists of accuracies
    #         train_score_array.append(knn.score(X_train, y_train))
    #         test_score_array.append(knn.score(X_test, y_test))
    #         #If the training score for the current k-value is better than the best k-value score, set the current k-value and its scores as the new best
    #         if (train_score_array[-1] > best_k_train_score):
    #             best_k = x
    #             best_k_train_score = train_score_array[-1]
    #             best_k_test_score = test_score_array[-1]
    #     #Creates 2 columns for aesthetics
    #     col3, col4 = st.columns([3,4])
    #     # First Column
    #     with col3:
    #         #Create a plot of the testing and training accuracies for each k-value
    #         fig, ax = plt.subplots(figsize=(12, 4))
    #         ax.plot(x_axis, train_score_array , label = "Train Score", c= "g")
    #         ax.plot(x_axis, test_score_array, label = "Test Score", c= "b")
    #         ax.set_title("What is our best k?")
    #         ax.set_xlabel('k')
    #         ax.set_ylabel('Accuracy')
    #         ax.legend()
    #         st.pyplot(fig)
    #     with col4:
    #         #Display k value, training accuracy, and testing accuracy of the best model, as well as how long it took to run the grid search
    #         st.subheader("Best k: " + str(int(best_k)))
    #         st.write("Best k Training Accuracy: " + str(np.round(100*best_k_train_score,2)) + "%")
    #         st.write("Best k Testing Accuracy: " + str(np.round(100*best_k_test_score,2)) + "%")
    #         endTime = time.time()
    #         st.write("Time to Run Grid Search: " + str(np.round(endTime-startTime,4)) + " Seconds")
