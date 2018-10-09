#!/usr/bin/env python
# coding: utf-8

import mlflow  #import mlflow
import pandas as pd  #Import pandas module 
from sklearn.preprocessing import LabelEncoder  #Module to LabelEncode
from sklearn.tree import DecisionTreeClassifier  #importing Decision Tree module 
import sys
from sklearn.metrics import accuracy_score     #Importing accuracy_score
from sklearn.model_selection import train_test_split   #Data set split cross validation 
import mlflow.sklearn



#Defining a function which removes null values from Age column and label encoding "Sex" column.
def Wrangle(df):
    #remove null values
    df=df.dropna(subset=['Age'])
    #encode categorical variables
    l=LabelEncoder()
    df['Sex']=l.fit_transform(df['Sex'])
    return df

if __name__=="__main__":

	df=pd.read_csv('titanic train.csv')
	df=Wrangle(df) #wrangling the data
	predictors=['Pclass','Age','Sex','Parch','SibSp'] #Choosing a set of predictors



	v2predictors=['Age','Sex','SibSp']
	v2X=df[v2predictors]  #subset of predictors
	v2Y=df['Survived']  #subset of labels
	v2_X_train, v2_X_test,v2_y_train, v2_y_test = train_test_split(v2X, v2Y, test_size=0.2, random_state=42)  #training and testing data prepared

        if (len(sys.argv) > 1):	
		splitter=str(sys.argv[1])
	else:
		splitter="best"

        if (len(sys.argv) > 2):	
		criterion=str(sys.argv[2])
	else:
		criterion="gini"



	with mlflow.start_run():
	    

	    print ("Training Model...V2")
	    clfv2 = DecisionTreeClassifier(splitter=splitter,criterion=criterion)#Creating an instance of Decision Tree Classifier 
	    v2=clfv2.fit(v2_X_train,v2_y_train) #Training the model 



	    print ("Predicting Values for V2")
	    v2_y_pred=v2.predict(v2_X_test) #Predictions for test Data
	    v2_Accuracy=accuracy_score(v2_y_pred,v2_y_test)*100.0 

	    print("V2 model (splitter=%s, criterion=%s):" % (splitter, criterion))
	    print("  V2 Accuracy: %s" % v2_Accuracy)

            mlflow.log_param("predictors", " ".join(v2predictors))
	    mlflow.log_param("splitter", splitter)
	    mlflow.log_param("criterion", criterion)
	    mlflow.log_metric("v2_accuracy", v2_Accuracy)
	    mlflow.sklearn.log_model(v2,"v2")   #Model can be logged as an Artifact 
            print ("Parameters logged and model saved for V2")





