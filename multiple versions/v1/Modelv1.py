#!/usr/bin/env python
# coding: utf-8


import mlflow  #import mlflow
import pandas as pd  #Import pandas module 
from sklearn.preprocessing import LabelEncoder  #Module to LabelEncode
from sklearn.tree import DecisionTreeClassifier  #importing Decision Tree module 
import sys
from sklearn.metrics import accuracy_score     #Importing accuracy_score
#from sklearn.cross_validation import train_test_split   #Data set split cross validation 
import mlflow.sklearn
from sklearn.model_selection import train_test_split
#Defining a function which removes null values from Age column and label encoding "Sex" colum
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


	v1predictors=['Pclass','Age','Parch']
	v1X=df[v1predictors]  #subset of predictors
	v1Y=df['Survived']  #subset of labels
	v1_X_train, v1_X_test,v1_y_train, v1_y_test = train_test_split(v1X, v1Y, test_size=0.2, random_state=42)  #training and testing data prepared
	
        if (len(sys.argv) > 1):	
		random_state=int(sys.argv[1])
	else:
		random_state=0

        if (len(sys.argv) > 2):	
		max_depth=int(sys.argv[2])
	else:
		max_depth=3


	
       

	with mlflow.start_run():
	
		
	    
	    print ("Training Model...V1")
	    clfv1 = DecisionTreeClassifier(random_state=random_state,max_depth=max_depth)#Creating an instance of Decision Tree Classifier 
	    v1=clfv1.fit(v1_X_train,v1_y_train) #Training the model 

	    print ("Predicting Values for V1")
	    v1_y_pred=v1.predict(v1_X_test) #Predictions for test Data
	    v1_Accuracy=accuracy_score(v1_y_pred,v1_y_test)*100.0 

	    print("V1 model (random_state=%f, max_depth=%f):" % (random_state, max_depth))
	    print("  V1 Accuracy: %s" % v1_Accuracy)
            mlflow.log_param("predictors", " ".join(v1predictors))
	    mlflow.log_param("random_state", random_state)
	    mlflow.log_param("max_depth", max_depth)
	    mlflow.log_metric("v1_accuracy", v1_Accuracy)
	    mlflow.sklearn.log_model(v1,"v1")   #Model can be logged as an Artifact 
	    print ("Parameters logged and model saved for V1")
	   




