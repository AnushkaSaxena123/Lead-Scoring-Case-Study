#!/usr/bin/env python
# coding: utf-8

# In[536]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[537]:


leads=pd.read_csv('Leads.csv')


# In[538]:


leadswithoutduplicate = leads.copy()

# Checking for duplicates and dropping the entire duplicate row if any
leadswithoutduplicate.drop_duplicates(subset=None, inplace=True)
leadswithoutduplicate.shape


# In[539]:


leadswithoutduplicate.info()


# In[540]:


# Percentage of null values of null value
round(100*(leadswithoutduplicate.isnull().sum())/len(leadswithoutduplicate.index),2)


# In[541]:


# Dropping columns with more than 40 percent missing values
leadswithoutduplicate.drop(columns=['Asymmetrique Activity Index','Asymmetrique Profile Index','Asymmetrique Activity Score','Asymmetrique Profile Score','Lead Quality'],axis=1,inplace=True)


# In[542]:


# Dropping prospect id and lead number as it is not important for the analysis
leadswithoutduplicate.drop(columns=['Lead Number'],axis=1,inplace=True)


# In[543]:


leadswithoutduplicate['Specialization'] = leadswithoutduplicate['Specialization'].fillna('not provided')
leadswithoutduplicate['City'] = leadswithoutduplicate['City'].fillna('not provided')
leadswithoutduplicate['Tags'] = leadswithoutduplicate['Tags'].fillna('not provided')
leadswithoutduplicate['What matters most to you in choosing a course'] = leadswithoutduplicate['What matters most to you in choosing a course'].fillna('not provided')
leadswithoutduplicate['What is your current occupation'] = leadswithoutduplicate['What is your current occupation'].fillna('not provided')
leadswithoutduplicate['Last Activity'] = leadswithoutduplicate['Last Activity'].fillna('not provided')
leadswithoutduplicate['Country'] = leadswithoutduplicate['Country'].fillna('not provided')
leadswithoutduplicate['TotalVisits'] = leadswithoutduplicate['TotalVisits'].fillna(np.NaN)
leadswithoutduplicate['Page Views Per Visit'] = leadswithoutduplicate['Page Views Per Visit'].fillna(np.NaN)
leadswithoutduplicate['Lead Source'] = leadswithoutduplicate['Lead Source'].fillna('not provided')


# In[544]:


leadswithoutduplicate['Last Activity'].value_counts(normalize=True)


# In[545]:


leadswithoutduplicate['Last Activity']=leadswithoutduplicate['Last Activity'].replace('not provided','Email Opened')


# In[546]:


leadswithoutduplicate['Country'].value_counts(normalize=True)


# In[547]:


leadswithoutduplicate['Country']=leadswithoutduplicate['Country'].replace('not provided','India')


# In[548]:


leadswithoutduplicate['City'].value_counts(normalize=True)


# In[549]:


leadswithoutduplicate['City']=leadswithoutduplicate['City'].replace('Select','not provided')


# In[550]:


leadswithoutduplicate['City'].value_counts(normalize=True)


# In[551]:


leadswithoutduplicate.drop(['City','Country'],axis=1,inplace=True)


# In[552]:


leadswithoutduplicate['Lead Origin'].value_counts(normalize=True)


# In[553]:


leadswithoutduplicate['Lead Source'].value_counts(normalize=True)


# In[554]:


leadswithoutduplicate['Lead Source']=leadswithoutduplicate['Lead Source'].replace('google','Google')


# In[555]:


leadswithoutduplicate['Lead Source']=leadswithoutduplicate['Lead Source'].replace('not provided','Google')


# In[556]:


leadswithoutduplicate['Do Not Email'].value_counts(normalize=True)


# In[557]:


leadswithoutduplicate['Do Not Call'].value_counts(normalize=True)


# In[558]:


leadswithoutduplicate['Specialization'].value_counts(normalize=True)


# In[559]:


leadswithoutduplicate['Specialization']=leadswithoutduplicate['Specialization'] = leadswithoutduplicate['Specialization'].replace(['Select','not provided'] ,'No Information')  


# In[560]:


leadswithoutduplicate['Specialization']=leadswithoutduplicate['Specialization'] = leadswithoutduplicate['Specialization'].replace(['Finance Management','Human Resource Management','Marketing Management','Operations Management','IT Projects Management','Supply Chain Management','Healthcare Management','Hospitality Management','Retail Management'] ,'Management Specializations')  


# In[561]:


leadswithoutduplicate['Specialization'].value_counts(normalize=True)


# In[562]:


leadswithoutduplicate['Last Notable Activity'].value_counts(normalize=True)


# In[563]:


leadswithoutduplicate['Last Notable Activity'] = leadswithoutduplicate['Last Notable Activity'].replace(['Had a Phone Conversation','Approached upfront','View in browser link Clicked','Email Received','Email Marked Spam','Visited Booth in Tradeshow','Resubscribed to emails','Form Submitted on Website'],'Others') 


# In[564]:


leadswithoutduplicate['What is your current occupation'].value_counts(normalize=True)


# In[565]:


leadswithoutduplicate['What is your current occupation'] = leadswithoutduplicate['What is your current occupation'].replace(['not provided'],'Unemployed') 


# In[566]:


leadswithoutduplicate['What matters most to you in choosing a course'].value_counts(normalize=True)


# In[567]:


leadswithoutduplicate['What matters most to you in choosing a course'] = leadswithoutduplicate['What matters most to you in choosing a course'].replace(['not provided'],'Better Career Prospects') 


# In[568]:


leadswithoutduplicate['Search'].value_counts(normalize=True)


# In[569]:


leadswithoutduplicate['Magazine'].value_counts(normalize=True)


# In[570]:


leadswithoutduplicate['Newspaper Article'].value_counts(normalize=True)


# In[571]:


leadswithoutduplicate['X Education Forums'].value_counts(normalize=True)


# In[572]:


leadswithoutduplicate['Digital Advertisement'].value_counts(normalize=True)


# In[573]:


leadswithoutduplicate['Through Recommendations'].value_counts(normalize=True)


# In[574]:


leadswithoutduplicate['Receive More Updates About Our Courses'].value_counts(normalize=True)


# In[575]:


leadswithoutduplicate['Tags'].value_counts(normalize=True)


# In[576]:


leadswithoutduplicate['Tags'] = leadswithoutduplicate['Tags'].replace(['switched off','Already a student','Not doing further education','invalid number','wrong number given','Interested  in full time MBA','In confusion whether part time or DLP', 'in touch with EINS','Diploma holder (Not Eligible)','Approached upfront','Graduation in progress','number not provided', 'opp hangup','Still Thinking','Lost to Others','Shall take in the next coming month','Lateral student','Interested in Next batch','Recognition issue (DEC approval)','Want to take admission but has financial problems','University not recognized'], 'Other_Tags')


# In[577]:


leadswithoutduplicate['Tags'] = leadswithoutduplicate['Tags'].replace(['not provided'], 'Not Specified')


# In[578]:


leadswithoutduplicate['Tags'].value_counts(normalize=True)


# In[579]:


leadswithoutduplicate['Update me on Supply Chain Content'].value_counts(normalize=True)


# In[580]:


leadswithoutduplicate['Get updates on DM Content'].value_counts(normalize=True)


# In[581]:


leadswithoutduplicate['Lead Profile'].value_counts(normalize=True)


# In[582]:


leadswithoutduplicate['I agree to pay the amount through cheque'].value_counts(normalize=True)


# In[583]:


leadswithoutduplicate['A free copy of Mastering The Interview'].value_counts(normalize=True)


# In[584]:


leadswithoutduplicate['Last Notable Activity'].value_counts(normalize=True)


# In[585]:


leadswithoutduplicate['How did you hear about X Education'].value_counts(normalize=True)


# In[586]:


leadswithoutduplicate.drop(['Magazine', 'Receive More Updates About Our Courses', 'Update me on Supply Chain Content', 
                          'Get updates on DM Content', 'I agree to pay the amount through cheque'], axis=1,inplace=True)


# In[587]:


leadswithoutduplicate['Lead Profile'].value_counts(normalize=True)


# In[588]:


#Dropping these columns as they contain 40% or more missing values
leadswithoutduplicate.drop(['Lead Profile','How did you hear about X Education'], axis=1,inplace=True)


# In[589]:


leadswithoutduplicate.info()


# In[590]:


plt.figure(figsize=(10,5))
b1=sns.countplot(leadswithoutduplicate['Lead Origin'], hue=leadswithoutduplicate.Converted)
b1.set_xticklabels(b1.get_xticklabels(),rotation=90)
for label in b1.containers:
    b1.bar_label(label)
plt.show()


# In[591]:


plt.figure(figsize=(15,10))
c1=sns.countplot(leadswithoutduplicate['Lead Source'], hue=leadswithoutduplicate.Converted)
c1.set_xticklabels(c1.get_xticklabels(),rotation=90)
for label in c1.containers:
    c1.bar_label(label)
plt.show()


# In[592]:


plt.figure(figsize=(10,5))
d1=sns.countplot(leadswithoutduplicate['Do Not Email'], hue=leadswithoutduplicate.Converted)
d1.set_xticklabels(d1.get_xticklabels(),rotation=90)
for label in d1.containers:
    d1.bar_label(label)
plt.show()


# In[593]:


plt.figure(figsize=(10,5))
e1=sns.countplot(leadswithoutduplicate['Do Not Call'], hue=leadswithoutduplicate.Converted)
e1.set_xticklabels(e1.get_xticklabels(),rotation=90)
for label in e1.containers:
    e1.bar_label(label)
plt.show()


# In[594]:


plt.figure(figsize=(10,5))
f1=sns.countplot(leadswithoutduplicate['Last Activity'], hue=leadswithoutduplicate.Converted)
f1.set_xticklabels(f1.get_xticklabels(),rotation=90)
for label in f1.containers:
    f1.bar_label(label)
plt.show()


# In[595]:


plt.figure(figsize=(10,5))
g1=sns.countplot(leadswithoutduplicate['Specialization'], hue=leadswithoutduplicate.Converted)
g1.set_xticklabels(g1.get_xticklabels(),rotation=90)
for label in g1.containers:
    g1.bar_label(label)
plt.show()


# In[596]:


plt.figure(figsize=(10,5))
j1=sns.countplot(leadswithoutduplicate['Search'], hue=leadswithoutduplicate.Converted)
j1.set_xticklabels(j1.get_xticklabels(),rotation=90)
for label in j1.containers:
    j1.bar_label(label)
plt.show()


# In[597]:


plt.figure(figsize=(10,5))
k1=sns.countplot(leadswithoutduplicate['Newspaper Article'], hue=leadswithoutduplicate.Converted)
k1.set_xticklabels(k1.get_xticklabels(),rotation=90)
for label in k1.containers:
    k1.bar_label(label)
plt.show()


# In[598]:


plt.figure(figsize=(10,5))
k1=sns.countplot(leadswithoutduplicate['X Education Forums'], hue=leadswithoutduplicate.Converted)
k1.set_xticklabels(k1.get_xticklabels(),rotation=90)
for label in k1.containers:
    k1.bar_label(label)
plt.show()


# In[599]:


plt.figure(figsize=(10,5))
m1=sns.countplot(leadswithoutduplicate['Digital Advertisement'], hue=leadswithoutduplicate.Converted)
m1.set_xticklabels(m1.get_xticklabels(),rotation=90)
for label in m1.containers:
    m1.bar_label(label)
plt.show()


# In[600]:


plt.figure(figsize=(10,5))
n1=sns.countplot(leadswithoutduplicate['Through Recommendations'], hue=leadswithoutduplicate.Converted)
n1.set_xticklabels(n1.get_xticklabels(),rotation=90)
for label in n1.containers:
    n1.bar_label(label)
plt.show()


# In[601]:


plt.figure(figsize=(15,10))
o1=sns.countplot(leadswithoutduplicate['Tags'], hue=leadswithoutduplicate.Converted)
o1.set_xticklabels(o1.get_xticklabels(),rotation=90)
for label in o1.containers:
    o1.bar_label(label)
plt.show()


# In[602]:


plt.figure(figsize=(10,5))
p1=sns.countplot(leadswithoutduplicate['A free copy of Mastering The Interview'], hue=leadswithoutduplicate.Converted)
p1.set_xticklabels(p1.get_xticklabels(),rotation=90)
for label in p1.containers:
    p1.bar_label(label)
plt.show()


# In[603]:


plt.figure(figsize=(10,5))
q1=sns.countplot(leadswithoutduplicate['Last Notable Activity'], hue=leadswithoutduplicate.Converted)
q1.set_xticklabels(q1.get_xticklabels(),rotation=90)
for label in q1.containers:
    q1.bar_label(label)
plt.show()


# In[604]:


plt.figure(figsize=(10,5))
i1=sns.countplot(leadswithoutduplicate['What matters most to you in choosing a course'], hue=leadswithoutduplicate.Converted)
i1.set_xticklabels(i1.get_xticklabels(),rotation=90)
for label in i1.containers:
    i1.bar_label(label)
plt.show()


# In[605]:


leadswithoutduplicate.drop(['Search','Through Recommendations','Newspaper','X Education Forums','Digital Advertisement','Newspaper Article'],axis=1,inplace=True)


# In[606]:


leadswithoutduplicate.drop(['Do Not Call'],axis=1,inplace=True)


# In[607]:


leadswithoutduplicate.drop(['What matters most to you in choosing a course'],axis=1,inplace=True)


# In[608]:


plt.figure(figsize=(10,5))
h1=sns.countplot(leadswithoutduplicate['What is your current occupation'], hue=leadswithoutduplicate.Converted)
h1.set_xticklabels(h1.get_xticklabels(),rotation=90)
for label in h1.containers:
    h1.bar_label(label)
plt.show()


# In[609]:


#Numerical Variables
plt.figure(figsize=(9,5))
r1=sns.barplot(y='TotalVisits', x='Converted',data=leadswithoutduplicate)
r1.set_xticklabels(r1.get_xticklabels(),rotation=360)
for label in r1.containers:
    r1.bar_label(label)
plt.show()


# In[610]:


plt.figure(figsize=(9,5))
t1=sns.barplot(y='Total Time Spent on Website', x='Converted',data=leadswithoutduplicate)
t1.set_xticklabels(t1.get_xticklabels(),rotation=360,ha='center')
for label in t1.containers:
    t1.bar_label(label)
plt.show()


# In[611]:


plt.figure(figsize=(9,5))
t1=sns.barplot(y='Page Views Per Visit', x='Converted',data=leadswithoutduplicate)
t1.set_xticklabels(t1.get_xticklabels(),rotation=360,ha='center')
for label in t1.containers:
    t1.bar_label(label)
plt.show()


# In[612]:


leadswithoutduplicate.describe([.25,.5,.75,.90,.95,.99])


# In[613]:


#Checking for outliers among numerical variables
plt.figure(figsize=(20, 25))
plt.subplot(4,3,1)
sns.boxplot(y = 'TotalVisits',  data = leadswithoutduplicate)
plt.subplot(4,3,2)
sns.boxplot(y = 'Total Time Spent on Website', data = leadswithoutduplicate)
plt.subplot(4,3,3)
sns.boxplot(y = 'Page Views Per Visit',  data = leadswithoutduplicate)
plt.show()


# In[614]:


leadswithoutduplicate['TotalVisits'] = leadswithoutduplicate['TotalVisits'].replace(np.NaN, leadswithoutduplicate['TotalVisits'].median())


# In[615]:


leadswithoutduplicate['Page Views Per Visit'] = leadswithoutduplicate['Page Views Per Visit'].replace(np.NaN, leadswithoutduplicate['Page Views Per Visit'].median())


# In[616]:


plt.figure(figsize = (6,6))
sns.heatmap(leadswithoutduplicate[['Converted','TotalVisits','Total Time Spent on Website','Page Views Per Visit']].corr(), annot = True, fmt='0.3g', cmap="YlGnBu")
plt.show()


# In[617]:


leadswithoutduplicate.info()


# In[618]:


LeadOrigin=pd.get_dummies(leadswithoutduplicate['Lead Origin'],prefix='LeadOrigin',drop_first=True)


# In[619]:


LeadSource=pd.get_dummies(leadswithoutduplicate['Lead Source'],prefix='LeadSource',drop_first=True)


# In[620]:


DoNotEmail=pd.get_dummies(leadswithoutduplicate['Do Not Email'],prefix='DoNotEmail',drop_first=True)


# In[621]:


LastActivity=pd.get_dummies(leadswithoutduplicate['Last Activity'],prefix='LastActivity',drop_first=True)


# In[622]:


Specialization1=pd.get_dummies(leadswithoutduplicate['Specialization'],prefix='Specialization1',drop_first=True)


# In[623]:


Whatisyourcurrentoccupation=pd.get_dummies(leadswithoutduplicate['What is your current occupation'],prefix='Whatisyourcurrentoccupation',drop_first=True)


# In[624]:


Tags1=pd.get_dummies(leadswithoutduplicate['Tags'],prefix='Tags1',drop_first=True)


# In[625]:


AfreecopyofMasteringTheInterview=pd.get_dummies(leadswithoutduplicate['A free copy of Mastering The Interview'],prefix='AfreecopyofMasteringTheInterview',drop_first=True)


# In[626]:


LastNotableActivity=pd.get_dummies(leadswithoutduplicate['Last Notable Activity'],prefix='LastNotableActivity',drop_first=True)


# In[627]:


leadswithoutduplicate=pd.concat([leadswithoutduplicate,LeadOrigin,LeadSource,DoNotEmail,LastActivity,Specialization1,Whatisyourcurrentoccupation,Tags1,AfreecopyofMasteringTheInterview,LastNotableActivity],axis=1)


# In[628]:


leadswithoutduplicate.drop(['Lead Origin','Lead Source','Do Not Email','Last Activity','Specialization','A free copy of Mastering The Interview','Last Notable Activity','What is your current occupation','Tags'],axis=1,inplace=True)


# In[629]:


leadswithoutduplicate.info()


# In[630]:


Y=leadswithoutduplicate['Converted']
Y.head()


# In[631]:


X=leadswithoutduplicate.drop(['Converted','Prospect ID'], axis=1)
X.head()


# In[632]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=100)


# In[633]:


scaler=MinMaxScaler()
num_vars=['TotalVisits','Total Time Spent on Website','Page Views Per Visit']
X_train[num_vars]=scaler.fit_transform(X_train[num_vars])
X_train.head()


# In[634]:


rfe=RFE(estimator=LogisticRegression())
rfe=rfe.fit(X_train,Y_train)


# In[635]:


rfe.support_


# In[636]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[637]:


col=X_train.columns[rfe.support_]


# In[638]:


X_train.columns[~rfe.support_]


# In[639]:


X_train_sm=sm.add_constant(X_train[col])
logm2=sm.GLM(Y_train,X_train_sm,family=sm.families.Binomial())
res=logm2.fit()
res.summary()


# In[640]:


Y_train_pred1=res.predict(X_train_sm)


# In[641]:


Y_train_pred_final1=pd.DataFrame({'Converted':Y_train.values,'Converted_Prob':Y_train_pred1})
Y_train_pred_final1['Prospect ID']=Y_train.index
Y_train_pred_final1.head()


# In[642]:


Y_train_pred_final1['predicted']=Y_train_pred_final1.Converted_Prob.map(lambda x:1 if x>0.5 else 0)
Y_train_pred_final1.head()


# In[643]:


confusion1=metrics.confusion_matrix(Y_train_pred_final1.Converted,Y_train_pred_final1.predicted)
print(confusion1)


# In[644]:


print(metrics.accuracy_score(Y_train_pred_final1.Converted,Y_train_pred_final1.predicted))


# In[645]:


vif=pd.DataFrame()
vif['Features']=X_train[col].columns
vif['VIF']=[variance_inflation_factor(X_train[col].values,i)for i in range(X_train[col].shape[1])]
vif['VIF']=round(vif['VIF'],2)
vif=vif.sort_values(by='VIF',ascending=False)
vif


# In[646]:


X_train_rfe1=X_train[col].drop(columns=['LeadSource_Facebook','LeadSource_NC_EDM','LeadSource_Reference','LeadSource_bing','LastActivity_Converted to Lead','LastActivity_Email Bounced','LastActivity_View in browser link Clicked','Specialization1_E-COMMERCE','Specialization1_International Business','Specialization1_Rural and Agribusiness','Specialization1_Services Excellence','Specialization1_Travel and Tourism','Whatisyourcurrentoccupation_Housewife','LastNotableActivity_Others'],axis=1)


# In[647]:


X_train_rfe2=sm.add_constant(X_train_rfe1)


# In[648]:


logm3=sm.GLM(Y_train,X_train_rfe2,family=sm.families.Binomial())
res=logm3.fit()
res.summary()


# In[649]:


Y_train_pred2=res.predict(X_train_rfe2)


# In[650]:


Y_train_pred_final2=pd.DataFrame({'Converted':Y_train.values,'Converted_Prob':Y_train_pred2})
Y_train_pred_final2['Prospect ID']=Y_train.index
Y_train_pred_final2.head()


# In[651]:


Y_train_pred_final2['predicted']=Y_train_pred_final2.Converted_Prob.map(lambda x:1 if x>0.5 else 0)
Y_train_pred_final2.head()


# In[652]:


confusion2=metrics.confusion_matrix(Y_train_pred_final2.Converted,Y_train_pred_final2.predicted)
print(confusion2)


# In[653]:


print(metrics.accuracy_score(Y_train_pred_final2.Converted,Y_train_pred_final2.predicted))


# In[654]:


vif1=pd.DataFrame()
vif1['Features']=X_train_rfe1.columns
vif1['VIF']=[variance_inflation_factor(X_train_rfe1.values,i)for i in range(X_train_rfe1.shape[1])]
vif1['VIF']=round(vif1['VIF'],2)
vif1=vif1.sort_values(by='VIF',ascending=False)
vif1


# In[655]:


X_train_rfe3=X_train_rfe1.drop(columns=['LeadOrigin_Quick Add Form','LastActivity_Olark Chat Conversation'],axis=1)


# In[656]:


X_train_rfe4=sm.add_constant(X_train_rfe3)


# In[657]:


logm4=sm.GLM(Y_train,X_train_rfe4,family=sm.families.Binomial())
res=logm4.fit()
res.summary()


# In[658]:


Y_train_pred3=res.predict(X_train_rfe4)


# In[659]:


Y_train_pred_final3=pd.DataFrame({'Converted':Y_train.values,'Converted_Prob':Y_train_pred3})
Y_train_pred_final3['Prospect ID']=Y_train.index
Y_train_pred_final3.head()


# In[660]:


Y_train_pred_final3['predicted']=Y_train_pred_final3.Converted_Prob.map(lambda x:1 if x>0.5 else 0)
Y_train_pred_final3.head()


# In[661]:


confusion3=metrics.confusion_matrix(Y_train_pred_final3.Converted,Y_train_pred_final3.predicted)
print(confusion3)


# In[662]:


print(metrics.accuracy_score(Y_train_pred_final3.Converted,Y_train_pred_final3.predicted))


# In[663]:


vif2=pd.DataFrame()
vif2['Features']=X_train_rfe3.columns
vif2['VIF']=[variance_inflation_factor(X_train_rfe3.values,i)for i in range(X_train_rfe3.shape[1])]
vif2['VIF']=round(vif2['VIF'],2)
vif2=vif2.sort_values(by='VIF',ascending=False)
vif2


# In[664]:


X_train_rfe5=X_train_rfe3.drop(columns=['LastNotableActivity_SMS Sent'],axis=1)


# In[665]:


X_train_rfe6=sm.add_constant(X_train_rfe5)


# In[666]:


logm5=sm.GLM(Y_train,X_train_rfe6,family=sm.families.Binomial())
res=logm5.fit()
res.summary()


# In[667]:


Y_train_pred4=res.predict(X_train_rfe6)


# In[668]:


Y_train_pred_final4=pd.DataFrame({'Converted':Y_train.values,'Converted_Prob':Y_train_pred4})
Y_train_pred_final4['Prospect ID']=Y_train.index
Y_train_pred_final4.head()


# In[669]:


Y_train_pred_final4['predicted']=Y_train_pred_final4.Converted_Prob.map(lambda x:1 if x>0.5 else 0)
Y_train_pred_final4.head()


# In[670]:


confusion4=metrics.confusion_matrix(Y_train_pred_final4.Converted,Y_train_pred_final4.predicted)
print(confusion4)


# In[671]:


#Predicted     Not Converted  Converted
#Actual 
#Not Converted  3833           169
#Converted      289            2177


# In[672]:


print(metrics.accuracy_score(Y_train_pred_final4.Converted,Y_train_pred_final4.predicted))


# In[673]:


vif3=pd.DataFrame()
vif3['Features']=X_train_rfe5.columns
vif3['VIF']=[variance_inflation_factor(X_train_rfe5.values,i)for i in range(X_train_rfe5.shape[1])]
vif3['VIF']=round(vif3['VIF'],2)
vif3=vif3.sort_values(by='VIF',ascending=False)
vif3


# In[674]:


TP = confusion4[1,1] # true positive 
TN = confusion4[0,0] # true negatives
FP = confusion4[0,1] # false positives
FN = confusion4[1,0] # false negatives


# In[675]:


Sensitivity=float(TP/(TP+FN))
print(Sensitivity)


# In[676]:


Specificity=float(TN/(TN+FP))
print(Specificity)


# In[677]:


# Defining the function to plot the ROC curve
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None

# Calling the function
draw_roc(Y_train_pred_final4.Converted, Y_train_pred_final4.Converted_Prob)


# In[678]:


numbers=[float(x)/10 for x in range(10)]
for i in numbers:
    Y_train_pred_final4[i]=Y_train_pred_final4.Converted_Prob.map(lambda x:1 if x>i else 0)
Y_train_pred_final4.head()


# In[679]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame(columns = ['Probability','Accuracy','Sensitivity','Specificity'])
num = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm4 = metrics.confusion_matrix(Y_train_pred_final4.Converted, Y_train_pred_final4[i] )
    total4=sum(sum(cm4))
    Accuracy = (cm4[0,0]+cm4[1,1])/total4   
    Specificity = cm4[0,0]/(cm4[0,0]+cm4[0,1])
    Sensitivity = cm4[1,1]/(cm4[1,0]+cm4[1,1])
    cutoff_df.loc[i] =[ i ,Accuracy,Sensitivity,Specificity]
print(cutoff_df)


# In[680]:


cutoff_df.plot.line(x='Probability', y=['Accuracy','Sensitivity','Specificity'])
plt.show()


# In[681]:


Y_train_pred_final4['final_Predicted'] = Y_train_pred_final4.Converted_Prob.map(lambda x: 1 if x > 0.3 else 0)
Y_train_pred_final4.head()


# In[682]:


confusion5=metrics.confusion_matrix(Y_train_pred_final4.Converted,Y_train_pred_final4.final_Predicted)
print(confusion5)


# In[683]:


print(metrics.accuracy_score(Y_train_pred_final4.Converted,Y_train_pred_final4.final_Predicted))


# In[684]:


TP1 = confusion5[1,1] # true positive 
TN2 = confusion5[0,0] # true negatives
FP1 = confusion5[0,1] # false positives
FN2 = confusion5[1,0] # false negatives


# In[685]:


Sensitivity1=float(TP1/(TP1+FN2))
print(Sensitivity1)


# In[686]:


Specificity1=float(TN2/(TN2+FP1))
print(Specificity1)


# In[687]:


from sklearn.metrics import precision_score,recall_score


# In[688]:


#Precision and Recall
Precision=float(TP/(TP+FP))
print(Precision)


# In[689]:


Recall=float(TP/(TP+FN))
print(Recall)


# In[690]:


precision_score(Y_train_pred_final4.Converted,Y_train_pred_final4.predicted)


# In[691]:


recall_score(Y_train_pred_final4.Converted,Y_train_pred_final4.predicted)


# In[692]:


from sklearn.metrics import precision_recall_curve


# In[693]:


Y_train_pred_final4.Converted,Y_train_pred_final4.predicted


# In[694]:


Y_train_pred_final4.Converted,Y_train_pred_final4.predicted
Precision,Recall,Thresholds=precision_recall_curve(Y_train_pred_final4.Converted,Y_train_pred_final4.Converted_Prob)


# In[695]:


plt.plot(Thresholds,Precision[:-1],'g-')
plt.plot(Thresholds,Recall[:-1],'r-')
plt.show()


# In[696]:


Y_train_pred_final4['finalpredicted1']=Y_train_pred_final4.Converted_Prob.map(lambda x:1 if x>0.38 else 0)
Y_train_pred_final4.head()


# In[697]:


confusion7=metrics.confusion_matrix(Y_train_pred_final4.Converted,Y_train_pred_final4.finalpredicted1)
print(confusion7)


# In[698]:


TP7 = confusion7[1,1] # true positive 
TN7 = confusion7[0,0] # true negatives
FP7 = confusion7[0,1] # false positives
FN7 = confusion7[1,0] # false negatives


# In[699]:


print(metrics.accuracy_score(Y_train_pred_final4.Converted,Y_train_pred_final4.finalpredicted1))


# In[700]:


Sensitivity7=float(TP7/(TP7+FN7))
print(Sensitivity7)


# In[701]:


Specificity7=float(TN7/(TN7+FP7))
print(Specificity7)


# In[702]:


precision_score(Y_train_pred_final4.Converted,Y_train_pred_final4.finalpredicted1)


# In[703]:


recall_score(Y_train_pred_final4.Converted,Y_train_pred_final4.finalpredicted1)


# In[704]:


X_test[num_vars]=scaler.transform(X_test[num_vars])
X_test.head()


# In[705]:


col=X_train_rfe5.columns


# In[706]:


print(col)


# In[707]:


X_test_rfe5=X_test[col]


# In[708]:


X_test_rfe6=sm.add_constant(X_test_rfe5)


# In[709]:


Y_test_pred4=res.predict(X_test_rfe6)


# In[710]:


Y_Pred_1=pd.DataFrame(Y_test_pred4)


# In[711]:


Y_test_df=pd.DataFrame(Y_test)


# In[712]:


Y_test_df['Prospect ID']=Y_test_df.index


# In[713]:


Y_Pred_1.reset_index(drop=True,inplace=True)
Y_test_df.reset_index(drop=True,inplace=True)


# In[714]:


Y_test_Pred_final=pd.concat([Y_test_df,Y_Pred_1],axis=1)


# In[715]:


Y_test_Pred_final.head()


# In[716]:


Y_test_Pred_final=Y_test_Pred_final.rename(columns={0:'Converted_Prob'})


# In[719]:


Y_test_Pred_final.head()


# In[720]:


Y_test_Pred_final['finalpredicted']=Y_test_Pred_final.Converted_Prob.map(lambda x:1 if x>0.38 else 0)


# In[721]:


Y_test_Pred_final.head()


# In[722]:


confusion10=metrics.confusion_matrix(Y_test_Pred_final.Converted,Y_test_Pred_final.finalpredicted)
print(confusion10)


# In[723]:


print(metrics.accuracy_score(Y_test_Pred_final.Converted,Y_test_Pred_final.finalpredicted))


# In[724]:


TP10 = confusion10[1,1] # true positive 
TN10 = confusion10[0,0] # true negatives
FP10 = confusion10[0,1] # false positives
FN10 = confusion10[1,0] # false negatives


# In[728]:


Sensitivity10=float(TP10/(TP10+FN10))
print(Sensitivity10)


# In[729]:


Specificity10=float(TN10/(TN10+FP10))
print(Specificity10)


# In[731]:


print('Precision ',precision_score(Y_test_Pred_final.Converted,Y_test_Pred_final.finalpredicted))
print('Recall ',recall_score(Y_test_Pred_final.Converted,Y_test_Pred_final.finalpredicted))


# In[732]:


draw_roc(Y_test_Pred_final.Converted,Y_test_Pred_final.finalpredicted)


# In[735]:


Precision,Recall,Thresholds=precision_recall_curve(Y_test_Pred_final.Converted,Y_test_Pred_final.Converted_Prob)


# In[736]:


plt.plot(Thresholds,Precision[:-1],'g-')
plt.plot(Thresholds,Recall[:-1],'r-')
plt.show()


# In[ ]:




