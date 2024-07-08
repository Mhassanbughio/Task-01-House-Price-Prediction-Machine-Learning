#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

#%matplotlib inline


# # Reading the Dataset 

# In[9]:


csv=pd.read_csv("USA Housing.csv")
csv


# # Data cleaning and preprocessing

# In[11]:


csv.head()


# In[65]:


csv.info()


# In[67]:


csv.shape


# In[73]:


csv.columns


# In[68]:


csv.isnull().sum()


# In[ ]:


#Data is already cleaned there is no null value in this data set


# # Visualization of Dataset

# In[14]:


sns.pairplot(csv)


# # Feature Engineering

# In[71]:


corr_mat = csv.corr()
corr_mat.style.background_gradient(cmap='BuGn')


# In[72]:


# Correlation matrix to understand feature relationships
correlation_matrix = csv.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Preprocessing: Selecting features and target variable
X = csv[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
       'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
       'parking', 'prefarea', 'furnishingstatus']]
y = csv['price']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[76]:


X=csv[['area', 'bedrooms', 'bathrooms', 'stories',
       'parking',]]
y=csv['price']


# ## Model selection and Training

# In[ ]:


X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=40, random_state=101)


# In[78]:


lm = LinearRegression()


# In[79]:


lm.fit(X_train,y_train)


# In[80]:


predictions = lm.predict(X_test)


# In[81]:


#print(predictions)


# In[82]:


plt.scatter(y_test,predictions, color="blue", label="prices")


# In[83]:


sns.displot((y_test,predictions),bins=50)


# # Model Evaluation

# In[84]:


from sklearn.metrics import mean_squared_error
# Calculate the mean squared error (MSE) of the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error (MSE): {mse}")


# In[85]:


from sklearn.metrics import mean_squared_error, r2_score
r2 = r2_score(y_test, predictions)
print("R-squared:", r2)


# In[ ]:





# In[ ]:




