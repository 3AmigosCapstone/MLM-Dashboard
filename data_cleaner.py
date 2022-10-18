#This Script allows the user to select what column they would like to predict, what columns they would like to use as features, and how they would like to split their data.
#It encodes the different features as binary or ordinal data depending on the feature.

#Import Libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

#Data Cleaning Function Definition, takes type of model (classification or regression) as a parameter.
def Data_Cleaning_Main(dataset,Model_Method):
    dataset_filename_dict = {'Mushroom':'data/mushrooms.csv','Car Evaluation':'data/car_evaluation.csv'}
    #Imports Data from CSV
    data = pd.read_csv(dataset_filename_dict[dataset])
    #Creates two empty lists to fill with different values depending on whether the model is regression or classification.
    prediction_columns =[]
    feature_columns = []

    if dataset == 'Mushroom':
        #Splits the columns into dataframes based on whether they are binary, ordinal, and categorical. (See https://archive.ics.uci.edu/ml/datasets/mushroom for data definitions)
        data_binary = data[['class','bruises','gill-size','stalk-shape','veil-type',]]
        data_ordinal = data[['gill-spacing','ring-number','population']]
        data_categorical = data.drop(columns=['class','bruises','gill-spacing','gill-size','stalk-shape','veil-type','ring-number','population'])

        #Classification and Regression Models require slightly different types of input and are slightly different in what they are able to predict (i.e. regression cannot predict non-binary non-ordinal data).
        #The following code creates a pair of predictable and feature-able columns depending on whether the model will be regression or classification.

        #Binary columns are treated the same for both regression and classification problems.
        #Adds each binary column header to both the featurable and predictable columns
        for u in data_binary.columns:
            feature_columns.append(u)
            prediction_columns.append(u)
        #Encodes lists of strings in Binary columns as binary data.
        data_binary.loc[:,('class')] = [1 if x=='e' else 0 for x in data_binary['class']]
        data_binary.loc[:,('bruises')] = [1 if x=='t' else 0 for x in data_binary['bruises']]
        data_binary.loc[:,('gill-size')] = [1 if x=='b' else 0 for x in data_binary['gill-size']]
        data_binary.loc[:,('stalk-shape')] = [1 if x=='e' else 0 for x in data_binary['stalk-shape']]
        data_binary.loc[:,('veil-type')] = [1 if x=='u' else 0 for x in data_binary['veil-type']]

        #Create a master dataframe of encoded data, starting as a copy of the binary data
        data_Encoded = data_binary.copy()
    elif dataset == 'Car Evaluation':
        #Create a master dataframe of encoded data, starting as a copy of the unencoded ordinal data
        data_ordinal = data.copy()
        data_Encoded = pd.DataFrame(index=data.index)

    #Adds data to master dataframe, and adds columns to predictable and featurable lists for classification problems
    if Model_Method == 'Classification':
        if dataset == 'Mushroom':
            #Encodes lists of strings in ordinal columns as ordinal data
            data_ordinal.loc[:,('gill-spacing')] = [0 if x=='d' else 1 if x=='c' else 2 for x in data_ordinal['gill-spacing']]
            data_ordinal.loc[:,('ring-number')] = [0 if x=='n' else 1 if x=='o' else 2 for x in data_ordinal['ring-number']]
            data_ordinal.loc[:,('population')] = [0 if x=='y' else 1 if x=='v' else 2 if x=='s' else 3 if x=='c' else 4 if x=='n' else 5 for x in data_ordinal['population']]
            #Adds categorical data to predicable lists as is, adds one hot encoded categorical data to featurable lists and master dataframe
            for u in data_categorical.columns:
                #Adds each category to predictable list
                prediction_columns.append(u)
                for v in set(data_categorical[u]):
                    #Performs One Hot Encoding on the Categorical Data, so each option for each category gets its own column of binary data, adds each binary column to list of featurable data
                    feature_columns.append(u+"_"+v)
                    data_Encoded.loc[:,(u+"_"+v)] = [1 if x == v else 0 for x in data_categorical[u]]
        elif dataset == 'Car Evaluation':
            #Encodes lists of strings in ordinal columns as ordinal data
            data_ordinal.loc[:,('class')] = [0 if x=='unacc' else 1 if x=='acc' else 2  if x == 'good' else 3 for x in data_ordinal['class']]
            data_ordinal.loc[:,('buying')] = [1 if x=='low' else 2 if x=='med' else 3  if x == 'high' else 4 for x in data_ordinal['buying']]
            data_ordinal.loc[:,('maint')] = [1 if x=='low' else 2 if x=='med' else 3  if x == 'high' else 4 for x in data_ordinal['maint']]
            data_ordinal.loc[:,('doors')] = [6 if x=='5more' else x for x in data_ordinal['doors']]
            data_ordinal.loc[:,('persons')] = [5 if x=='more' else x for x in data_ordinal['persons']]
            data_ordinal.loc[:,('lug_boot')] = [1 if x=='small' else 1 if x=='med' else 2 for x in data_ordinal['lug_boot']]
            data_ordinal.loc[:,('safety')] = [1 if x=='low' else 2 if x=='med' else 3 for x in data_ordinal['safety']]
        #Adds each column as is to the predictable and featurable lists, as well as to the master dataframe
        for u in data_ordinal.columns:
            feature_columns.append(u)
            prediction_columns.append(u)
            data_Encoded.loc[:,([u])] = data_ordinal.loc[:,([u])]
    #Adds data to master dataframe, and adds columns to predictable and featurable lists for regression problems
    elif Model_Method == 'Regression':
        #Treats both ordinal and categorical data as categorical, and adds one hot encoded categorical data to predictable and featureable lists, as well as to the master dataframe.
        for u in data_ordinal.columns:
            for v in set(data_ordinal[u]):
                feature_columns.append(u+"_"+v)
                prediction_columns.append(u+"_"+v)
                data_Encoded.loc[:,(u+"_"+v)] = [1 if x == v else 0 for x in data_ordinal[u]]
        if dataset == 'Mushroom':
            for u in data_categorical.columns:
                for v in set(data_categorical[u]):
                    feature_columns.append(u+"_"+v)
                    prediction_columns.append(u+"_"+v)
                    data_Encoded.loc[:,(u+"_"+v)] = [1 if x == v else 0 for x in data_categorical[u]]

    dataset_classname_dict = {'Mushroom':'Poisonous vs. Edible','Car Evaluation':'Car Acceptability'}

    #Creates three columns for aesthetic purposes
    col1, col2, col3 = st.columns(3)

    #First column
    with col1:
        if Model_Method == 'Classification' or dataset == 'Mushroom':
            #Allows the User to Select whether they would like to predict edibility, or some other column
            target_selector = st.radio('What Would You Like To Predict?',[dataset_classname_dict[dataset],'Other'])
        else:
            target_selector = 'Other'
    #Second Column
    with col2:
        #If the User chose to predict edibility, make the target 'class' and let the user select whether they want importance features, all features, or a custom set of features
        if target_selector == dataset_classname_dict[dataset] :
            target = 'class'
            feature_selector = st.radio('What Features Would You Like to Include?',['All Features','Other'])
        #If the User chose not to predict edibility, allow them to choose a target from the list of predictable columns, and allow them to choose whether they want to use all features or a custom set of features
        else:
            target = st.selectbox('Pick a Target:',prediction_columns)
            feature_selector = st.radio('What Features Would You Like to Include?',['All Features','Other'])
    #Third Column
    with col3:
        #If the User chose All features, make the features the entire list of featurable data that isn't pulled directly from the target column
        if feature_selector == 'All Features':
            features = [x for x in feature_columns if x[:-2] not in target]
        #If the User chose a custom set of features, allow them to choose anything from the list of featurable data that isn't pulled directly from the target column
        else:
            features = st.multiselect('Pick Your Features:',[x for x in feature_columns if x[:-2] not in target])

        #If the user hasn't selected any features, break the script
        if len(features)==0:
            st.warning("Please Select at least one feature:")
            st.stop()

        #If the User chose a classification model, make the target come from the raw string data
        if Model_Method == 'Classification':
            Y = data[target]
        #If the User chose a regression model, make the target come for the encoded numerical data
        else:
            Y = data_Encoded[target]
        #Make the features from from whichever columns the User selected that aren't pulled directly from teh target column
        X = data_Encoded[[x for x in features if x[:-2] not in target]]

    #Allow the user to select how they want to split their data, default to 30%
    split_float = st.number_input('Select Test/Train Split Percentage:',min_value = .01,max_value=.99, value = .30, step = .01)
    #Split data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 0, test_size = split_float)
    #Return a tuple of different dataframes
    return X_train, X_test, y_train, y_test
