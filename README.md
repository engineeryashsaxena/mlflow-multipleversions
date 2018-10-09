# mlflow-multipleversions

This repository is having a folder "multiple version" which is a mlflow project . 
In that folder , there are 2 more folders "v1" and "v2" which are different projects containing different models having different hyperparameters and predictors )

The functionality demonstrated here is switching between the models . 
Since this is a mlflow project ,the command to execute the folder is : 

      mlflow run multiple versions  

By default it takes 'v1' as the model . To select a particular model ,the parameter is 'model_version' and command should be:

      mlflow run multiple versions -Pmodel_version=1
      
The possible values for model_version can be 1 and 2 for model v1 and v2 respectively.
Note that model_version is also logged as a parameter. 

You can run different models and switch between them using the above command . Also, the logs are shown on UI ( Screenshot is also present in the repository ) 


#Further Reading 

Each Model is also translated into a MLflow project . Details are as follows:

Details of Model V1 :
  -> Predictors are : 
  -> Hyperparameters are : random_split ( default value is 0)  ;  max_depth ( default value is 3) 
  
Details of Model V2 :
  -> Predictors are : 
  -> Hyperparameters are : splitter (default value is "best" ) ; criterion (default value is "gini") 
  
 Since these are two independent project they can also be run using command :
 
          mlflow run <folder_name> 
 
          mlflow run <folder_name> -P{hyperparameter}={hyper_parameter_value}
 
 example : For Model v1 , setting parameters random_split as 23 and max _depth as 10 , command will be :
 
          mlflow run v1 -Prandom_split=23 -Pmax_depth=10
