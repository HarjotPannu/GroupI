#!/usr/bin/env python
# coding: utf-8

# # import necessary libraries

# In[3]:


import itertools
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression


# # import dataset

# In[4]:


#Read the data
df = pd.read_csv('C:\\Users\\dell\\Desktop\\Capstone_Project\\news.csv')


# In[5]:


df


# In[6]:


# Get shape and head
df.shape
df.head()


# In[7]:


df.rename(columns={'Unnamed: 0': 'Numbers','title': 'Headlines','text': 'Body'}, inplace = True)


# In[8]:


df


# In[9]:


for col in df.columns: 
    print(col) 


# In[10]:


df = df.drop(['Numbers'], axis = 1)
df = df.dropna()


# In[11]:


df.describe()


# In[12]:


# Get the lables
labels = df.label
labels.head()


# In[13]:


df.label.value_counts()


# In[14]:


X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values


# In[15]:


X[0]


# In[16]:


Y[0]


# In[17]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
mat_body = cv.fit_transform(X[:,1]).todense()


# In[18]:


mat_body


# In[23]:


cv_head = CountVectorizer(max_features = 5000)
mat_head = cv_head.fit_transform(X[:,0]).todense()


# In[24]:


mat_head


# In[25]:


X_mat = np.hstack((mat_head, mat_body))


# In[26]:


df.isnull().sum()


# In[27]:


plt.figure(figsize=(9,5))
sns.countplot(df.label)


# In[28]:


df.info


# In[29]:


df.corr()


# In[30]:


# Split the dataset into training and testing datasets
x_train,x_test,y_train,y_test = train_test_split(df['Headlines'], labels, test_size=0.2, random_state=7)


# In[31]:


print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


# In[32]:


sns.countplot(Y)


# In[39]:


from sklearn.metrics import confusion_matrix,classification_report


# In[40]:


x_train,x_test,y_train,y_test = train_test_split(df['Headlines'], labels, test_size=0.2, random_state=7)


# In[41]:


# Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform training set, transform testing set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[42]:


# Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
# Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[43]:


# Build confusion matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])

