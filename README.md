# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: sugeshan s
RegisterNumber: 212224040337

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
dataset=pd.read_csv('student_scores.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)

```

## Output:
TO READ HEAD AND TAIL VALUES

<img width="157" height="126" alt="image" src="https://github.com/user-attachments/assets/bdce4349-7825-47a6-a267-7f3de8cf2fe1" />

<img width="192" height="130" alt="319848309-ea9cb89a-f4b8-473d-84b8-1f92b2ee64a2" src="https://github.com/user-attachments/assets/a4f5fcce-9f9e-4cbe-aeeb-7561a9333718" />

Compare Dataset

<img width="631" height="487" alt="319848327-9d7409fe-cb21-4727-9bd1-365c85ab9f1a" src="https://github.com/user-attachments/assets/9037a381-8872-4a46-a6c0-3e89f7f5c55d" />

Predicted Value

<img width="756" height="72" alt="319848353-44b39961-15f3-4ef1-b64f-01bd7c4fff12" src="https://github.com/user-attachments/assets/57e0075a-cae2-471f-9dbf-ed0d7697e61f" />

Graph For Training Set

<img width="793" height="566" alt="319852122-be5ed2ff-790c-4d0d-84c0-a230f9f4d2df" src="https://github.com/user-attachments/assets/3fab46d9-222e-493e-9204-6e655d427b01" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
