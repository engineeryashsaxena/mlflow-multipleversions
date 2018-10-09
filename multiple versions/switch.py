#!/usr/bin/env python
# coding: utf-8


import sys
import mlflow.projects 
import mlflow

if __name__=="__main__":
	param=sys.argv
        if len(param)>1:
		mlflow.log_param("model_version",int(param[1]))   #version specified by user
	else: 
		mlflow.log_param("model_version",1)   #default value 
	if int(param[1])==1:
		print "Version 1 executing..."
		mlflow.projects.run('v1')   #Running model v1 
	elif int(param[1])==2:
		print "Version 2 executing..."
		mlflow.projects.run('v2')   #Running model v2
	else:   
		mlflow.log_param("model_version","Error in choosing model")   #Error in choosing model 
		print "Wrong version selected"

	
	
	
	   




