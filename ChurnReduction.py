
# coding: utf-8

# In[5]:


#Load libraries
import os
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns
from random import randrange, uniform
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.cross_validation import train_test_split


# In[6]:


#Set working directory
os.chdir("C:/Users/SHRAVYA/Desktop/edwisor/project 2")


# In[7]:


#Load data
Train_data= pd.read_csv("Train_data.csv")
Test_data = pd.read_csv("Test_data.csv")


# In[8]:


#----------------------------------PRE PROCESSING-EXPLORATORY DATA ANALYSIS----------------------------------------------------#
#As we can see, few variables have a wrongly placed datatype.
#The variable , phone.number is actually meant to be a continuoos variable but it is analysed as a factor variable with 3333 levels. hence it has to be altered
#The variable area.code has to be a factor variable.
Train_data['area code']=Train_data['area code'].astype(object)
Test_data['area code']=Test_data['area code'].astype(object)



# In[5]:


#---------------------------------------MISSING VALUE ANALYSIS-----------------------------------------------------------------#
#Create dataframe with missing percentage
missing_val = pd.DataFrame(Train_data.isnull().sum())

#Reset index
missing_val = missing_val.reset_index()

#Rename variable
missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'Missing_percentage'})

#Calculate percentage
missing_val['Missing_percentage'] = (missing_val['Missing_percentage']/len(Train_data))*100

#descending order
missing_val = missing_val.sort_values('Missing_percentage', ascending = False).reset_index(drop = True)

#save output results 
missing_val.to_csv("Missing_perc.csv", index = False)
Train_data.dtypes
missing_val
Train_data.isnull().sum()
Test_data.isnull().sum()


# In[9]:


#As we can see, no missing values are present. Hence no missing analysis is required.
Train_data.dtypes


# In[7]:


#Plot boxplot to visualize Outliers
get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(Train_data['account length'])


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(Train_data['number vmail messages'])


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(Train_data['total day minutes'])


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(Train_data['total day calls'])


# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(Train_data['total day charge'])


# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(Train_data['total eve minutes'])


# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(Train_data['total eve calls'])


# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(Train_data['total eve charge'])


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(Train_data['total night minutes'])


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(Train_data['total night calls'])


# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(Train_data['total night charge'])


# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(Train_data['total intl minutes'])


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(Train_data['total intl calls'])


# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(Train_data['total intl charge'])


# In[21]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(Train_data['number customer service calls'])


# In[22]:


#save numeric names
cnames =  ["account length", "number vmail messages", "total day minutes", "total day calls", "total day charge", "total eve minutes", "total eve calls", "total eve charge",
           "total night minutes", "total night calls", "total night charge","total intl minutes","total intl calls","total intl charge","number customer service calls"]


# In[23]:


#Detect and delete outliers from data
for i in cnames:
     print(i)
     q75, q25 = np.percentile(Train_data.loc[:,i], [75 ,25])
     iqr = q75 - q25

     min = q25 - (iqr*1.5)
     max = q75 + (iqr*1.5)
     print(min)
     print(max)
    
     Train_data = Train_data.dropTrain_data[Train_data.loc[:,i] < min].index)
     Train_data = Train_data.drop(Train_data[Train_data.loc[:,i] > max].index)


# In[24]:


##-----------------------------------------------Correlation analysis------------------------------------------------------
#Correlation plot
df_corr = Train_data.loc[:,cnames]


# In[25]:


#Set the width and hieght of the plot
f, ax = plt.subplots(figsize=(7, 5))

#Generate correlation matrix
corr = df_corr.corr()

#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# In[26]:


#Chisquare test of independence
#Save categorical variables
cat_names = ["state", "area code", "phone number", "international plan", "voice mail plan"]


# In[27]:


#loop for chi square values
for i in cat_names:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(Train_data['Churn'], Train_data[i]))
    print(p)


# In[28]:


Train_data.dtypes
Test_data.dtypes


# In[10]:


Test_data = Test_data.drop(['state','area code', 'total day minutes', 'total eve minutes', 'total night minutes', 'total intl minutes','phone number'], axis=1)


# In[11]:


Train_data = Train_data.drop(['state','area code', 'total day minutes', 'total eve minutes', 'total night minutes', 'total intl minutes','phone number'], axis=1)


# In[12]:


Train_data['international plan'] = Train_data['international plan'].astype('category')
Train_data['voice mail plan'] = Train_data['voice mail plan'].astype('category')
Train_data['Churn'] = Train_data['Churn'].astype('category')
Test_data['international plan'] = Test_data['international plan'].astype('category')
Test_data['voice mail plan'] = Test_data['voice mail plan'].astype('category')
Test_data['Churn'] = Test_data['Churn'].astype('category')


# In[13]:


Train_data['international plan'] = (Train_data['international plan'] == ' yes').astype(int)
Train_data['voice mail plan'] = (Train_data['voice mail plan'] == ' yes').astype(int)
Train_data['Churn'] = (Train_data['Churn'] == ' True.').astype(int)
Test_data['international plan'] = (Test_data['international plan'] == ' yes').astype(int)
Test_data['voice mail plan'] = (Test_data['voice mail plan'] == ' yes').astype(int)
Test_data['Churn'] = (Test_data['Churn'] == ' True.').astype(int)


# In[15]:


#Divide data into train and test
X = Train_data.values[:, 0:13]
Y = Train_data.values[:,13]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2)


# In[35]:


#Decision Tree
C50_model = tree.DecisionTreeClassifier(criterion='entropy').fit(X_train, y_train)

#predict new test cases
C50_Predictions = C50_model.predict(X_test)

#Create dot file to visualise tree  #http://webgraphviz.com/
# dotfile = open("pt.dot", 'w')
# df = tree.export_graphviz(C50_model, out_file=dotfile, feature_names = marketing_train.columns)


# In[36]:


#build confusion matrix
# from sklearn.metrics import confusion_matrix 
# CM = confusion_matrix(y_test, y_pred)
CM = pd.crosstab(y_test, C50_Predictions)

#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model
#accuracy_score(y_test, y_pred)*100
((TP+TN)*100)/(TP+TN+FP+FN)
#accuracy= 90.55%
#False Negative rate=19.0 
(FN*100)/(FN+TP)


# In[37]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier

RF_model = RandomForestClassifier(n_estimators = 20).fit(X_train, y_train)


# In[38]:


RF_Predictions = RF_model.predict(X_test)


# In[39]:


#build confusion matrix
# from sklearn.metrics import confusion_matrix 
# CM = confusion_matrix(y_test, y_pred)
CM = pd.crosstab(y_test, RF_Predictions)

#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model
#accuracy_score(y_test, y_pred)*100
((TP+TN)*100)/(TP+TN+FP+FN)

#False Negative rate 
(FN*100)/(FN+TP)

#Accuracy: 94.75%
FNR: 28.26


# In[46]:


#KNN implementation
from sklearn.neighbors import KNeighborsClassifier

KNN_model = KNeighborsClassifier(n_neighbors = 9).fit(X_train, y_train)


# In[47]:


#predict test cases
KNN_Predictions = KNN_model.predict(X_test)


# In[50]:


#build confusion matrix
CM = pd.crosstab(y_test, KNN_Predictions)

#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model
#accuracy_score(y_test, y_pred)*100
((TP+TN)*100)/(TP+TN+FP+FN)

#False Negative rate 
(FN*100)/(FN+TP)

#Accuracy: 87.10
#FNR: 97


# In[51]:


#Naive Bayes
from sklearn.naive_bayes import GaussianNB

#Naive Bayes implementation
NB_model = GaussianNB().fit(X_train, y_train)


# In[52]:


#predict test cases
NB_Predictions = NB_model.predict(X_test)


# In[56]:


#Build confusion matrix
CM = pd.crosstab(y_test, NB_Predictions)

#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model
#accuracy_score(y_test, y_pred)*100
((TP+TN)*100)/(TP+TN+FP+FN)

#False Negative rate 
#(FN*100)/(FN+TP)

#Accuracy: 86.95
#FNR: 64.36


# In[1]:


#logistic regression
import numpy as np 
import matplotlib.pyplot as plt


# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()


# In[18]:


logisticRegr.fit(X_train, y_train)


# In[20]:


logisticRegr.predict(X_test[0:13])


# In[21]:


predictions = logisticRegr.predict(X_test)


# In[22]:


score = logisticRegr.score(X_test, y_test)
print(score)
#0.8740629685157422


# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


# In[24]:


cm = metrics.confusion_matrix(y_test, predictions)
print(cm)
#FNR = 79.01

