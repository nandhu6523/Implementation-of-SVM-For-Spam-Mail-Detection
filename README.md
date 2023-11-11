# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages.
2.Import the dataset to operate on.
3.Split the dataset.
4.Predict the required output.
5.End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Nandhini S
RegisterNumber: 212222220028 
*/
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
Result:

 ![Screenshot 2023-11-11 141602](https://github.com/nandhu6523/Implementation-of-SVM-For-Spam-Mail-Detection/assets/123856724/c2ae8a3e-62ad-459d-9f47-e5890380536b)

Data.head():
 ![Screenshot 2023-11-11 141619](https://github.com/nandhu6523/Implementation-of-SVM-For-Spam-Mail-Detection/assets/123856724/8e1a0390-e9d1-4a4b-9fcf-925a9d400a30)

 Data.info():
 ![Screenshot 2023-11-11 141630](https://github.com/nandhu6523/Implementation-of-SVM-For-Spam-Mail-Detection/assets/123856724/2124ce7b-b2ff-49d2-bae9-887fad621783)

data isnull().sum():
 ![Screenshot 2023-11-11 141644](https://github.com/nandhu6523/Implementation-of-SVM-For-Spam-Mail-Detection/assets/123856724/9724df92-113a-41f5-a30b-3f06c6bf99ca)

Y_prediction value:
  ![Screenshot 2023-11-11 141655](https://github.com/nandhu6523/Implementation-of-SVM-For-Spam-Mail-Detection/assets/123856724/7e825c3d-2eaf-4b35-8fe0-cd617e63bbb5)

Accuract value:
 ![Screenshot 2023-11-11 141708](https://github.com/nandhu6523/Implementation-of-SVM-For-Spam-Mail-Detection/assets/123856724/150a95ad-8acd-4872-b942-a8fd5f4ad8bb)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
