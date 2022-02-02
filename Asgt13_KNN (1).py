#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# ### Data

# In[2]:


# Loading dataset
data = pd.read_csv('C:/Users/17pol/Downloads/glass.csv')


# ### EDA & Data preprocessing

# In[3]:


data.shape


# In[4]:


data.head()


# In[5]:


data.sample(10)


# In[6]:


data.info()


# In[7]:


data.Type.unique()


# In[8]:


data.isna().sum()


# In[9]:


data.describe()


# In[10]:


data['Type'].value_counts()


# In[11]:


sns.countplot(x = 'Type', data = data)


# In[15]:


#Heat map to see difference between parameters
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap ='coolwarm' )


# In[13]:


# plotting pairplot

sns.pairplot(data)


# In[17]:


# Visualizing the content of different elements in the various types of glass

sns.stripplot(x='Type',y='RI',data=data)


# In[18]:


sns.stripplot(x='Type',y='Na',data=data)


# In[19]:


sns.stripplot(x='Type',y='Mg',data=data)


# In[20]:


sns.stripplot(x='Type',y='Al',data=data)


# In[21]:


sns.stripplot(x='Type',y='Si',data=data)


# In[22]:


sns.stripplot(x='Type',y='K',data=data)


# In[23]:


sns.stripplot(x='Type',y='Ca',data=data)


# In[24]:


sns.stripplot(x='Type',y='Ba',data=data)


# In[25]:


sns.stripplot(x='Type',y='Fe',data=data)


# In[26]:


data.columns


# In[27]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[28]:


scaler.fit(data.drop('Type', axis = 1))


# In[29]:


scaled_features=scaler.transform(data.drop('Type',axis=1))
data_head=pd.DataFrame(scaled_features,columns=data.columns[:-1])
data_head


# In[30]:


# Splitting data into test data and train data

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data_head,data['Type'], test_size=0.3, random_state=42)


# In[31]:


print('Shape of x_train: ', x_train.shape)
print('Shape of x_test: ', x_test.shape)
print('Shape of y_train: ', y_train.shape)
print('Shape of y_test: ', y_test.shape)


# ### Building KNN model

# In[32]:


model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)


# In[33]:


pred = model.predict(x_test)
pred


# In[34]:


pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': pred})
pred_df


# In[35]:


kfold = KFold(n_splits=10)
results = cross_val_score(model, x_train, y_train, cv=kfold)
print(results.mean())


# In[36]:


from sklearn.metrics import classification_report,confusion_matrix
print('Classification Report ',classification_report(y_test,pred))


# In[37]:


# Printing confusion matrix
print('Confusion Matrix\n',confusion_matrix(y_test,pred))


# In[38]:


error_rate=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_i=knn.predict(x_test)
    error_rate.append(np.mean(pred_i!=y_test))


# In[39]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('error_rate vs. K Value')
plt.xlabel('K')
plt.ylabel('error_rate')


# In[40]:


import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
# choose k between 1 to 41
k_range = range(1, 41)
k_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train, cv=5)
    k_scores.append(scores.mean())
# plot to see clearly
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


# In[41]:


#We can see that at K=1 we have low error rate and high value of accuracy. Hence we will perform the test with K=1
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)


# In[42]:


pred=knn.predict(x_test)
pred


# In[43]:


pred_df = pd.DataFrame({'Actual' : y_test, 'Predicted' : pred})
pred_df


# In[44]:


kfold = KFold(n_splits=10)
results = cross_val_score(knn, x_train, y_train, cv=kfold)
print(results.mean())


# In[45]:


from sklearn.metrics import classification_report,confusion_matrix
print('Classification Report ',classification_report(y_test,pred))


# In[46]:


# Printing confusion matrix
print('Confusion Matrix\n',confusion_matrix(y_test,pred))


# In[47]:


#We can see that accuracy is improved when K=1
sns.countplot(pred)


# In[ ]:




