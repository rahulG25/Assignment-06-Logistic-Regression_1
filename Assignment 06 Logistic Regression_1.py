#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[4]:


# Importing the dataset
bank=pd.read_csv('bank-full.csv')
bank


# # EDA

# In[5]:


bank.info()


# In[6]:


# One-Hot Encoding of categrical variables
data1=pd.get_dummies(bank,columns=['job','marital','education','contact','poutcome'])
data1


# In[7]:


# To see all columns
pd.set_option("display.max.columns", None)
data1


# In[9]:


data1.info()


# In[10]:


# Custom Binary Encoding of Binary o/p variables 
data1['default'] = np.where(data1['default'].str.contains("yes"), 1, 0)
data1['housing'] = np.where(data1['housing'].str.contains("yes"), 1, 0)
data1['loan'] = np.where(data1['loan'].str.contains("yes"), 1, 0)
data1['y'] = np.where(data1['y'].str.contains("yes"), 1, 0)
data1


# In[11]:


# Find and Replace Encoding for month categorical varaible
data1['month'].value_counts()


# In[12]:


order={'month':{'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}}


# In[13]:


data1=data1.replace(order)


# In[14]:


data1


# In[15]:


data1.info()


# # Model Building

# In[17]:


# Dividing our data into input and output variables
x=pd.concat([data1.iloc[:,0:11],data1.iloc[:,12:]],axis=1)
y=data1.iloc[:,11]


# In[18]:


# Logistic regression model
classifier=LogisticRegression()
classifier.fit(x,y)


# # Model Predictions

# In[19]:


# Predict for x dataset
y_pred=classifier.predict(x)
y_pred


# In[20]:


y_pred_df=pd.DataFrame({'actual_y':y,'y_pred_prob':y_pred})
y_pred_df


# # Testing Model Accuracy

# In[21]:


# Confusion Matrix for the model accuracy
confusion_matrix = confusion_matrix(y,y_pred)
confusion_matrix


# In[22]:


# The model accuracy is calculated by (a+d)/(a+b+c+d)
(39107+1282)/(39107+815+4007+1282)


# In[23]:


# As accuracy = 0.8933, which is greater than 0.5; Thus [:,1] Threshold value>0.5=1 else [:,0] Threshold value<0.5=0 
classifier.predict_proba(x)[:,1] 


# In[24]:


# ROC Curve plotting and finding AUC value
fpr,tpr,thresholds=roc_curve(y,classifier.predict_proba(x)[:,1])
plt.plot(fpr,tpr,color='red')
auc=roc_auc_score(y,y_pred)

plt.plot(fpr,tpr,color='red',label='logit model(area  = %0.2f)'%auc)
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.show()

print('auc accuracy:',auc)


# The model with highest roc_auc_acore is consider as a best model. From above Logistic regression model have highest roc_auc_score 0.6087663555959076.
