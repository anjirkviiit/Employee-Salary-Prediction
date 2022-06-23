import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

hiringData=pd.read_csv("hiring.csv")

#Fetching column wise null values
hiringData.isnull().sum()

# Data consist null values so we need to apply Feature Engineering on the data set
#In hiring data we have null values so we need to perform some feature engineering techniques
hiringData['experience'].fillna(0,inplace=True)

#In hiring data we have numerical values so we need to impute mean of the column
hiringData['test_score'].fillna(hiringData['test_score'].mean(),inplace=True)

#the column experiance consist of categorical values so we need to convert into numerical values
def convertToInt(value):
    value_Dict={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'zero':0,0:0}
    return value_Dict[value]
#Label encoding
hiringData['experience']=hiringData['experience'].apply(lambda x:convertToInt(x))
#Spliting dataset
X=hiringData.iloc[:,:3].copy()
y=hiringData.iloc[:,-1].copy()

regressorModel=LinearRegression()
regressorModel.fit(X,y)
#Saving model to pickle file
pickle.dump(regressorModel, open('model.pkl','wb'))

#loading the model
model=pickle.load(open('model.pkl','rb'))
#Testing the model
print(model.predict([[4,10,6]]))