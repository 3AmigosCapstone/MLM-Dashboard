#This script allows the user to build and test a support vector classification model

#Import Libraries
import streamlit as st
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

#Import custom data cleaning function from data_cleaner.py script
from data_cleaner import *

#Support Vector Machine Main Function
def SVM_Main(dataset):
    #Page Title
    st.header("Support Vector Classification")
    st.write("---")
    #Allows user to build a model
    st.header("Build Your Model")
    #Allows user to select target, features, and split data
    X_train, X_test, y_train, y_test = Data_Cleaning_Main(dataset,'Classification')
    st.write("___")
    #Allows user to train a model
    st.header("Train Your Model")
    #Creates 4 columns for aesthetics
    col1, col2, col3, col4 = st.columns([2,2,2,1])
    #First Column
    with col1:
        #Allow user to select a kernel
        kernel = st.selectbox("Choose a kernel:", ['rbf','linear','poly','sigmoid'])
        #If the user chose a polynomial kernel, allow them to choose the degree, otherwise set the degree to 0
        if kernel == 'poly':
            degree = st.number_input('Choose a Degree:',min_value=2,value=3,step=1)
        else:
            degree = 0
    #Second Column
    with col2:
        #Allow the user to chose a c value, defaults to 1
        c_value = st.number_input("Choose a C value:",min_value = .01, value = 1.0, step = .01)
    #Third Column
    with col3:
        #Allow the user to chose a gamma value, defaults to 1/n_features
        gamma_value = st.number_input("Choose a gamma value:",min_value = .001, value = 1.0/len(list(X_train.columns)), step = .001)
    #Fourth Column
    with col4:
        #spacers
        for i in range(2):
            st.write("")
            #Once the user presses this button, it will train and test the model
        train_button = st.button("Train Model")
    #Train and test the model
    if train_button:
        #Get start time
        startTime = time.time()
        #Fit SVC Model
        svclassifier = SVC(kernel=kernel, C=c_value,gamma=gamma_value)
        svclassifier.fit(X_train, y_train)
        #Display the training and testing accuracy of the model
        st.subheader("Training Accuracy: " + str(np.round(100*svclassifier.score(X_train, y_train),2)) + "%")
        st.subheader("Testing Accuracy: " + str(np.round(100*svclassifier.score(X_test, y_test),2)) + "%" )
        #Show user how long the model took to train and test
        endTime = time.time()
        st.write("Time to Run: " + str(np.round(endTime-startTime,4)) + " Seconds")
    st.write("___")
    #Allow user to grid search their linear_model
    st.header("Grid Search Your Model")
    st.write("*Note: Grid Search on SVC takes a considerable amount of time*")
    #Create 4 columns for aesthetics
    col5, col6, col7, col8 = st.columns([2,2,2,1])
    #First Column
    with col5:
        # Create empty list of kernels
        kernelList = []
        #Allow user to select kernels
        st.write("Select Kernel(s):")
        for x in ['rbf','linear','poly','sigmoid']:
            #Add any kernel the user selects to the list of kernels
            if st.checkbox(x,value=True):
                kernelList.append(x)
        #If the user selects poly as one of the kernels, allow them to select a range of degrees, otherwise there is only one degree (0)
        if 'poly' in kernelList:
            #Allow the user to select the minimum and maximum degrees
            degree_min = st.number_input('Minimum Degree: ',min_value=2,value=3,step=1)
            degree_max = st.number_input('Maximum Degree: ',min_value = int(degree_min + 1), value = int(degree_min + 1), step = 1)
            # Add each value in between max and min degrees to a list of degrees
            degrees = []
            degree_i = degree_min
            while degree_i <= degree_max:
                degrees.append(degree_i)
                degree_i += 1
        else:
            degrees = [0]
    #Second Column
    with col6:
        #Allow user to select c values. Let them choose a minimum, maximum, and the size of the interval in between steps
        c_interval = st.number_input("Select C Value Interval Size:",min_value = 0.01, value = 1.0)
        c_min = st.number_input('Minimum C: ',min_value=0.01,value=1.0,step = 0.01)
        c_max = st.number_input('Maximim C: ',min_value=c_min,value=c_min + c_interval,step = 0.01)
        #If the max and min are at least one interval apart, create a list of c values that contains as many intervals as possible between the max and min
        if c_max-c_min >= c_interval:
            c_values = []
            c_i = c_min
            while c_i <= c_max:
                c_values.append(c_i)
                c_i += c_interval
    #Third Column
    with col7:
        #Allow user to select gamma values. Let them choose a minimum, maximum, and the size of the interval in between steps
        gamma_interval = st.number_input("Select Gamma Value Interval Size:",min_value = 0.01, value = 1.0)
        gamma_min = st.number_input('Minimum Gamma: ',min_value=0.01,value=1.0,step = 0.01)
        gamma_max = st.number_input('Maximim Gamma: ',min_value=gamma_min,value=gamma_min + gamma_interval,step = 0.01)
        #If the max and min are at least one interval apart, create a list of c values that contains as many intervals as possible between the max and min
        if gamma_max-gamma_min >= gamma_interval:
            gamma_values = []
            gamma_i = gamma_min
            while gamma_i <= gamma_max:
                gamma_values.append(gamma_i)
                gamma_i += gamma_interval
    #Fourth Column
    with col8:
        #Make sure the data works by ensuring that there is at least one kernel, and that the intervals, maximums, and minimums work together
        if len(kernelList) == 0:
            st.warning("Please Select at least one kernel")
        elif c_max-c_min < c_interval:
            st.warning("C Minimum and Maxmimum must contain at least one interval")
        elif gamma_max-gamma_min < gamma_interval:
            st.warning("Gamma Minimum and Maxmimum must contain at least one interval")
        #If the selected options are all valid, create a button that will run the grid search when the user clicks it
        else:
            grid_search_button = st.button("Run Grid Search")
    #Run the Grid search
    if grid_search_button:
        #get start time
        startTime = time.time()
        #Create a parameter grid based on the users selections
        param_grid = {'kernel' : kernelList, 'degree':degrees,'C':c_values,'gamma':gamma_values}
        #Perform the grid search
        grid_search = GridSearchCV(SVC(), param_grid, return_train_score = True)
        grid_search.fit(X_train, y_train)
        #Get the best parameters for the grid search
        best_params = grid_search.best_params_
        #Display the best parameters to the user
        st.subheader("Best Kernel: " + best_params['kernel'])
        if best_params['kernel'] == 'poly':
            st.subheader("Best Degree: " + str(best_params['degree']))
        st.subheader("Best C Value: " + str(best_params['C']))
        st.subheader("Best Gamma Value: " + str(best_params['gamma']))
        #Display the Training and testing accuracy of the best model to the user
        st.subheader("Training Accuracy:" + str(np.round(100*grid_search.score(X_train, y_train),2)) + "%")
        st.subheader("Testing Accuracy:" + str(np.round(100*grid_search.score(X_test, y_test),2)) + "%" )
        #Display how long it took to run the grid search
        endTime = time.time()
        st.write("Time to Run Grid Search: " + str(np.round(endTime-startTime,4)) + " Seconds")
