#!/usr/bin/env python
# coding: utf-8

# # Spam SMS Detection

# In[ ]:


# Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Load the dataset using pandas.read_csv().
df = pd.read_csv("spam.csv",encoding='latin-1')


# In[4]:


df


# In[5]:


# Display the first few rows of the dataframe.
df.head()


# In[6]:


# Display the last few rows of the dataframe.
df.tail()


# In[7]:


# Checking the shape of the dataset
df.shape


# In[8]:


# Checking the columns of the dataset
df.columns


# In[9]:


df.info()


# In[10]:


# Checking for any missing values in the dataset
df.isnull().sum()


# In[11]:


# Display the summary statistics of the datase
df.describe()


# In[12]:


df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)


# In[13]:


df.isnull().sum()


# In[14]:


df.rename(columns={'v1':'target', 'v2':'text'}, inplace=True)
df.head()


# In[23]:


df.info()


# In[24]:


from sklearn.preprocessing import OneHotEncoder
one_hot_encoded = pd.get_dummies(df ,columns = ['target','text'], drop_first=True)
one_hot_encoded


# In[25]:


# Select the features and the target variable
x=df[['text']]
y=df['target']


# In[26]:


# Split the dataset into training and testing sets using train_test_split from sklearn.model_selection.
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2,random_state=2)


# In[27]:


from sklearn.feature_extraction.text import TfidfVectorizer


vectorizer = TfidfVectorizer()

xtrain_vec = vectorizer.fit_transform(xtrain['text'])
xtest_vec = vectorizer.transform(xtest['text'])


# In[28]:


# Import the Logistic Regression class from sklearn.linear_model.
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[29]:


model.fit(xtrain_vec,ytrain)


# In[30]:


model.score(xtrain_vec,ytrain)


# In[31]:


# Display the model's coefficients and intercept.
print('cofficient:', model.coef_)
print('intercept:', model.intercept_)


# In[ ]:




