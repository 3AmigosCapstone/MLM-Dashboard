#This script is pure infrastructure, it is the central script controlling the dashboard

#Import Libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

#Import functions from other scripts
from Sidebar import *
from Analysis_Tabs.DTC_Tab import *
from Analysis_Tabs.kNN_Tab import *
from Analysis_Tabs.LR_Tab import *
from Analysis_Tabs.NB_Tab import *
from Analysis_Tabs.SVM_Tab import *

#Main Page Function Defintion
def Mushroom_Main():

    #Page Layout
    st.set_page_config(layout="wide")

    #Allow the user to choose what dataset to work with
    dataset = st.radio("Select a Dataset:",['Mushroom','Car Evaluation'])

    if dataset == 'Mushroom':
        #Page Title and Subtitle lol
        st.title("*To Eat or Not To Eat:*")
        st.header("A Mushroom Story")
    else:
        #Page Title and Subtitle lol
        st.title("*To Buy or Not To Buy:*")
        st.header("A Car Story")

    #Calls a function that comes from the Sidebar.py script that gives us credit
    Sidebar_Main()

    #This creates an input where the user can select what type of machine learning model they can use
    Model_Type = st.selectbox("What type of model would you like to build?",['k-Nearest Neighbors','Naive Bayes','Support Vector Machine','Decision Tree','Logistic Regression'])

    #This takes the user's input and runs the function that generates the interface that the user needs to build the selected machine learning model
    if Model_Type == 'k-Nearest Neighbors':
        #Calls a function that comes from the kNN_Tab.py script for creating a k-Nearest Neighbors Classification Model
        kNN_Main(dataset)
    elif Model_Type == 'Naive Bayes':
        #Calls a function that comes from the NB_Tab.py script for creating a Naive Bayes Classification Model
        NB_Main(dataset)
    elif Model_Type == 'Support Vector Machine':
        #Calls a function that comes from the SVM_Tab.py script for creating a Support Vector Classification Model
        SVM_Main(dataset)
    elif Model_Type == 'Decision Tree':
        #Calls a function that from the DTC_Tab.py script for creating a Decision Tree Classification Model
        DTC_Main(dataset)
    elif Model_Type == 'Logistic Regression':
        #Calls a function that comes from the LR_Tab.py script for creating a Logistic Regression Model
        LR_Main(dataset)

#Calls the function above to show everything
Mushroom_Main()
