#!/usr/bin/env python
# coding: utf-8

# ## Implement a KNN model to classify the animals in to categories
# 

# In[1]:


# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# In[3]:


# Loading dataset
data = pd.read_csv('C:/Users/17pol/Downloads/Zoo.csv')

print(data.shape)

data.head()


# In[4]:


data.sample(10)


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.isna().sum()


# In[8]:


# Checking how many types of animals are present in dataset
data['type'].value_counts()


# In[9]:


sns.countplot(x = 'type', data = data,)


# In[10]:


# Lets plot to check how many animals are domestic
plt.figure(figsize=(6,4))
data.domestic.value_counts().plot(kind="bar")
plt.xlabel('Is Domestic')
plt.ylabel("Count")
plt.plot()


# In[11]:


pd.crosstab(data.type, data.domestic)


# In[12]:


# Species wise domestic and non-domestic animals
pd.crosstab(data.type, data.domestic).plot(kind="bar", figsize=(10, 5), title="Class wise Domestic & Non-Domestic Count")
plt.plot()


# In[13]:


# Lets see how many animals provides us milk
data.milk.value_counts()


# In[14]:


pd.crosstab(data.type, data.milk)


# In[15]:


pd.crosstab(data.type, data.milk).plot(kind="bar", title="Class wise Milk providing animals", 
                                                         figsize=(10, 5))


# In[16]:


# Lets see how many animals live under water. i.e aquatic
data.aquatic.value_counts()


# In[17]:


data[data.aquatic==1].type.value_counts()


# In[18]:


# Lets plot category wise animals having fins
pd.crosstab(data.type, data.aquatic).plot(kind="bar", figsize=(10, 5))


# In[19]:


data_temp = data
data_temp = data_temp.groupby(by='animal name').mean()
plt.rcParams['figure.figsize'] = (16,10) 
sns.heatmap(data_temp, cmap="inferno")
ax = plt.gca()
ax.set_title("Features for the Animals")


# In[20]:


data_temp = data_temp.groupby(by='type').mean()
plt.rcParams['figure.figsize'] = (16,10) 
sns.heatmap(data_temp, annot=True, cmap="inferno")
ax = plt.gca()
ax.set_title("HeatMap of Features for the Classes")


# In[21]:


# We will be removing column animal_name as it does not help us in classification
data1 = data.drop('animal name', axis = 1)
data1.head()


# In[22]:


# Splitting data into X and y
X = data1.drop('type', axis = 1)
y = data['type']
X


# In[24]:


y


# In[25]:


# Split X and y into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 0)


# In[26]:


print('Shape of x_train: ', X_train.shape)
print('Shape of x_test: ', X_test.shape)
print('Shape of y_train: ', y_train.shape)
print('Shape of y_test: ', y_test.shape)


# ### Building KNN Model

# In[27]:


# Fit k-nearest neighbors classifier with training sets for n = 5
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)


# In[28]:


# Run prediction
y_pred = knn.predict(X_test)


# In[29]:


y_pred


# In[30]:


pred_df = pd.DataFrame({'Actual' : y_test, 'Predicted' : y_pred})
pred_df


# In[31]:


from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,y_pred))


# In[32]:


print(confusion_matrix(y_test,y_pred))


# In[35]:


sns.set_style('whitegrid')

plt.rcParams['figure.figsize'] = (10, 6) 
_, ax = plt.subplots()
ax.hist(y_test, color = 'r', alpha = 0.5, label = 'actual', bins=7)
ax.hist(y_pred, color = 'b', alpha = 0.5, label = 'prediction', bins=7)
ax.yaxis.set_ticks(np.arange(0,11))
ax.legend(loc = 'best')
plt.show()


# In[34]:


# Get score for different values of n
k_list = np.arange(1, 50, 2)
mean_scores = []
accuracy_list = []
error_rate = []

for i in k_list:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    score = cross_val_score(knn,X_train, y_train,cv=3)
    mean_scores.append(np.mean(score))
    error_rate.append(np.mean(pred_i != y_test))

print("Mean Scores:")
print(mean_scores)
print("Error Rate:")
print(error_rate)


# ### Visualizing model performance for different numbers of K

# In[36]:


# Plot n values and average accuracy scores
plt.plot(k_list,mean_scores, marker='o')

# Added titles and adjust dimensions
plt.title('Accuracy of Model for Varying Values of K')
plt.xlabel("Values of K")
plt.ylabel("Mean Accuracy Score")
plt.xticks(k_list)
plt.rcParams['figure.figsize'] = (12,12) 

plt.show()


# In[37]:


# Plot n values and average accuracy scores
plt.plot(k_list,error_rate, color='r', marker = 'o')

# Added titles and adjust dimensions
plt.title('Error Rate for Model for Varying Values of K')
plt.xlabel("Values of K")
plt.ylabel("Error Rate")
plt.xticks(k_list)
plt.rcParams['figure.figsize'] = (12, 6) 

plt.show()


# In[38]:


data1.columns


# In[39]:


data1['has_legs'] = np.where(data1['legs']>0,1,0)
data1 = data1[['hair','feathers','eggs','milk', 'airborne', 'aquatic', 'predator', 'toothed', 'backbone', 'breathes',
               'venomous','fins','legs','has_legs','tail','domestic','catsize','type']]
data1.head()


# In[40]:


# Select columns to add to X and y sets
features = list(data1.columns.values)
features.remove('legs')
features.remove('type')
X2 = data1[features]
y2 = data1['type']
# Split X and y into train and test
X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y2,random_state = 0)
# Fit k-nearest neighbors classifier with training sets for n = 5
knn2 = KNeighborsClassifier(n_neighbors = 5)
knn2.fit(X2_train, y2_train)
KNeighborsClassifier()
# Run prediction
y2_pred = knn2.predict(X2_test)
print(confusion_matrix(y2_test,y2_pred))


# In[41]:


print(classification_report(y2_test,y2_pred))


# In[42]:


plt.rcParams['figure.figsize'] = (9,9) 
_, ax = plt.subplots()
ax.hist(y2_test, color = 'm', alpha = 0.5, label = 'actual', bins=7)
ax.hist(y2_pred, color = 'c', alpha = 0.5, label = 'prediction', bins=7)
ax.yaxis.set_ticks(np.arange(0,11))
ax.legend(loc = 'best')

plt.show()


# In[43]:


# Get score for different values of n
k_list = np.arange(1, 50, 2)
mean_scores2 = []
accuracy_list2 = []
error_rate2 = []

for i in k_list:
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(X2_train,y2_train)
    pred_i = knn2.predict(X2_test)
    score = cross_val_score(knn2,X2_train, y2_train,cv=3)
    mean_scores2.append(np.mean(score))
    error_rate2.append(np.mean(pred_i != y2_test))

print("Mean Scores:")
print(mean_scores)
print("Error Rate:")
print(error_rate)


# In[44]:


# Plot n values and average accuracy scores
plt.plot(k_list,mean_scores, color='b',marker='o', label='Model using Number of Legs')
plt.plot(k_list,mean_scores2, color='m',marker='x', label='Model using Presence of Legs')

# Added titles and adjust dimensions
plt.title('Accuracy of Model for Varying Values of K')
plt.xlabel("Values of K")
plt.ylabel("Mean Accuracy Score")
plt.xticks(k_list)
plt.legend()
plt.rcParams['figure.figsize'] = (12,12) 

plt.show()


# In[45]:


# Plot n values and average accuracy scores
plt.plot(k_list,error_rate, color='r', marker = 'o', label='Model using Number of Legs')
plt.plot(k_list,error_rate2, color='c', marker = 'x', label='Model using Presence of Legs')

# Added titles and adjust dimensions
plt.title('Error Rate for Model for Varying Values of K')
plt.xlabel("Values of K")
plt.ylabel("Error Rate")
plt.xticks(k_list)
plt.legend()
plt.rcParams['figure.figsize'] = (12,6) 

plt.show()


# ### Inference
#  Replacing the feature legs has improved the accuracy of KNN models at every value where n >3. This may be due to the model taking the number of legs as a continuous, numeric data point rather than as an integer.

# In[ ]:





# In[ ]:




