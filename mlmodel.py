#import neccessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

data = pd.read_csv("Top_Crop.csv")
data.head()

 data.isnull().sum()

 data.info()

 data.columns

data["Commodity"].value_counts()

data["District_Name"].value_counts()

sns.distplot(x=data["Demand"])

sns.distplot(x=data["Max Price"])

sns.countplot(x=data["Commodity"])

 sns.countplot(x=data["District_Name"])

 sns.barplot(data=data,x = "Year", y = "Demand",hue = "Commodity")

 crop = sns.lineplot(data=data, x='Max Price', y='Demand', color='#4c934c')
crop.set_title('Price V/S Demand')

data.plot(x = 'Year',y = 'Demand',style = 'o' )
plt.title('Year v/s Demand')
plt.xlabel('Year')
plt.ylabel('Demand')
plt.show()

data.plot(x = 'Max Price',y = 'Demand',style = 'o' )
plt.title('Price v/s Demand')
plt.xlabel('Price')
plt.ylabel('Demand')
plt.show()

 plt.figure(figsize=(8,4))
sns.heatmap(data.corr(),annot=True,fmt="0.1f")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["District_Name"] = le.fit_transform(data["District_Name"])
data.head()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Commodity"] = le.fit_transform(data["Commodity"])
data.head()

x=data.iloc[:,:-1]
y=data.iloc[:,-1]
x.head()

x=data.iloc[:,:-1]
y=data.iloc[:,-1]
x.head()

xtrain.shape

ytrain.shape

xtest.shape


 ytest.shape

 from sklearn.linear_model import LinearRegression
data=LinearRegression()


 data.fit(xtrain,ytrain)

data.coef_

data.intercept_

ypred = data.predict(xtest)
ypred

 data.score(xtest,ytest)


data.score(xtest,ytest)*100,"%"