#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn import linear_model

import warnings
warnings.filterwarnings('ignore')


# In[2]:


all_coaching_changes = pd.read_csv("Desktop/SAAS/final_coaching_changes_1.csv")
pd.set_option("display.max_rows", None, "display.max_columns", None)
all_coaching_changes


# In[3]:


features = all_coaching_changes.drop(['Unnamed: 0', 'Season', 'Team', 'W', 'L', 'Coaches'], axis = 1)


features["W/L Classify"] = np.where(features["Change in W/L%"] <= 0, 0, 1)

features


# In[4]:


corrMatrix = features.corr()
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(corrMatrix, annot=True, ax=ax)
plt.show()


# In[5]:


features_copy = features.copy() 
features_copy = features_copy.drop(columns = ["Prev_OeFG%", "Prev_DeFG%", "Prev_NRtg", "Prev_SRS"], axis = 1)

corrMatrix = features_copy.corr()
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(corrMatrix, annot=True, ax=ax)


# In[6]:


from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

y = features_copy['W/L Classify']
X = features_copy.drop(columns = ['W/L%','Change in W/L%', 'W/L Classify', 'Midseason Hire', 'Roster Changes', 'Coach Experience', 'Average Age', 'Young Players', 'Prev_ORB%', 'Prev_OFT/FGA%', 'Prev_DTOV%', 'Prev_DRB%'], axis= 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[7]:


kfold = KFold(n_splits=10, random_state=0, shuffle=True)
kfold


# In[8]:


print(y_train)


# In[9]:


logreg_model = LogisticRegression(solver='liblinear')
logreg_model.fit(X_train, y_train)
results = cross_val_score(logreg_model, X_train, y_train, cv=kfold, error_score="raise")
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[10]:


results


# In[11]:


y_pred = logreg_model.predict(X_test)
print(y_pred)


# In[12]:


y_pred


# In[13]:


y_test


# In[14]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n", cm)


# In[15]:


from sklearn.metrics import accuracy_score


# In[16]:


accuracy_score(y_test, y_pred)


# In[17]:


group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                cm.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cm.flatten()/np.sum(cm)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm, annot=labels, fmt='', cmap='Reds')


# In[18]:


import statsmodels.api as sm
log_reg_summary = sm.Logit(y_train, X_train).fit()
print(log_reg_summary.summary())


# In[19]:


features_copy.plot.scatter(x = 'Change in W/L%', y = 'W/L%')


# In[20]:


features_copy.plot.scatter(x = 'Roster Changes', y = 'W/L%')


# In[21]:


features_copy.plot.scatter(x = 'Coach Experience', y = 'W/L%')


# In[22]:


features_copy.plot.scatter(x = 'Coach Career W/L%', y = 'W/L%')


# In[23]:


features_copy.plot.scatter(x = 'Average Age', y = 'W/L%')


# In[24]:


features_copy.plot.scatter(x = 'Prev_ORtg', y = 'W/L%')


# In[25]:


features_copy.plot.scatter(x = 'Prev_DRtg', y = 'W/L%')


# In[26]:


features_copy.plot.scatter(x = 'Prev_Pace', y = 'W/L%')


# In[27]:


features_copy.plot.scatter(x = 'Prev_OTOV%', y = 'W/L%')


# In[28]:


features_copy.plot.scatter(x = 'Prev_ORB%', y = 'W/L%')


# In[29]:


features_copy.plot.scatter(x = 'Prev_OFT/FGA%', y = 'W/L%')


# In[30]:


features_copy.plot.scatter(x = 'Prev_DTOV%', y = 'W/L%')


# In[31]:


features_copy.plot.scatter(x = 'Prev_DRB%', y = 'W/L%')


# In[32]:


features_copy.plot.scatter(x = 'Prev_DFT/FGA%', y = 'W/L%')


# In[33]:


features_copy.plot.scatter(x = 'Prev_W/L%', y = 'W/L%')


# In[34]:


lin_y = features_copy['W/L%']
lin_X = (features_copy.drop(columns = ['W/L%', 'Change in W/L%', 'Young Players', 'W/L Classify', 'Midseason Hire', 'Coach Experience', 'Prev_Pace', 'Prev_OTOV%', 'Prev_ORB%', 'Prev_OFT/FGA%', 'Prev_DTOV%', 'Prev_DRB%', 'Prev_DFT/FGA%'], axis= 1))
lin_X_Coeffs = (features_copy.drop(columns = ['Change in W/L%', 'Young Players', 'W/L Classify', 'Midseason Hire', 'Coach Experience', 'Prev_Pace', 'Prev_OTOV%', 'Prev_ORB%', 'Prev_OFT/FGA%', 'Prev_DTOV%', 'Prev_DRB%', 'Prev_DFT/FGA%'], axis= 1))
lin_X


# In[35]:


linX_train, linX_test, liny_train, liny_test = train_test_split(lin_X, lin_y, test_size=0.3, random_state=0)

linX_test


# In[36]:


from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

linreg.fit(linX_train, liny_train)

linreg.score(linX_test, liny_test)


# In[37]:


model = sm.OLS(liny_train,linX_train)
results = model.fit()
print(results.summary())


# In[38]:


midseason_sorted = all_coaching_changes.sort_values("Midseason Hire")
midseason_sorted


# In[39]:


import statistics

fullseason_hires = []
midseason_hires = []

for i in range(len(midseason_sorted)):
    wl = midseason_sorted["Change in W/L%"].loc[i]
    hire = midseason_sorted["Midseason Hire"].loc[i]
    if hire:
        midseason_hires.append(wl)
    else:
        fullseason_hires.append(wl)
        
print(statistics.mean(fullseason_hires) * 100, len(fullseason_hires))
print(statistics.mean(midseason_hires) * 100, len(midseason_hires))


# In[40]:


fullseason_mean = statistics.mean(fullseason_hires)
fullseason_var = statistics.stdev(fullseason_hires) ** 2
fullseason_n = len(fullseason_hires)


print("Full-Season Mean:", fullseason_mean)
print("Full-Season Variance:", fullseason_var)
print("Full-Season n =", fullseason_n)


# In[41]:


midseason_mean = statistics.mean(midseason_hires)
midseason_var = statistics.stdev(midseason_hires) ** 2
midseason_n = len(midseason_hires)


print("Mid-Season Mean:", midseason_mean)
print("Mid-Season Variance:", midseason_var)
print("Mid-Season n =", midseason_n)


# In[42]:


import math

t_value = (abs(fullseason_mean - midseason_mean))/(math.sqrt((fullseason_var/fullseason_n) + (midseason_var/midseason_n)))
print("t statistic is", t_value)


# In[43]:


bins = np.linspace(-0.75, 0.75)

plt.hist(fullseason_hires, bins, alpha=0.5, label='Full-Season Hires')
plt.hist(midseason_hires, bins, alpha=0.5, label='Mid-Season Hires')
plt.legend(loc='upper right')
plt.show()


# In[44]:


midseason_sorted = midseason_sorted.reset_index()


# In[45]:


midseason_sorted


# In[46]:


all_fullseason_hires = midseason_sorted.iloc[0:277]
all_fullseason_hires


# In[47]:


all_midseason_hires = midseason_sorted.iloc[277:]
all_midseason_hires


# In[48]:


fullseason_features = all_fullseason_hires.drop(['index', 'Unnamed: 0', 'Season', 'Team', 'W', 'L', 'Coaches', 'Midseason Hire'], axis = 1)
midseason_features = all_midseason_hires.drop(['index', 'Unnamed: 0', 'Season', 'Team', 'W', 'L', 'Coaches', 'Midseason Hire'], axis = 1)


fullseason_features["W/L Classify"] = np.where(fullseason_features["Change in W/L%"] <= 0, 0, 1)
midseason_features["W/L Classify"] = np.where(midseason_features["Change in W/L%"] <= 0, 0, 1)

fullseason_features
midseason_features


# In[49]:


fullseason_corrMatrix = fullseason_features.corr()
fullseason_features_fig, fullseason_features_ax = plt.subplots(figsize=(20,20))
sns.heatmap(fullseason_corrMatrix, annot=True, ax=fullseason_features_ax)
plt.show()


# In[50]:


fullseason_features_copy = fullseason_features.copy() 
fullseason_features_copy = fullseason_features_copy.drop(columns = ["Prev_NRtg", "Prev_SRS", "Prev_OeFG%", "Prev_DeFG%"], axis = 1)

fullseason_corrMatrix = fullseason_features_copy.corr()
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(fullseason_corrMatrix, annot=True, ax=ax)


# In[51]:


y = fullseason_features_copy['W/L Classify']
X = fullseason_features_copy.drop(columns = ['W/L%','Change in W/L%', 'W/L Classify', 'Roster Changes', 'Coach Experience', 'Average Age', 'Prev_Pace', 'Prev_ORB%', 'Prev_OFT/FGA%', 'Prev_DTOV%', 'Prev_DRB%'], axis= 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[52]:


kfold = KFold(n_splits=10, random_state=0, shuffle=True)
kfold


# In[53]:


logreg_model = LogisticRegression(solver='liblinear')
logreg_model.fit(X_train, y_train)
results = cross_val_score(logreg_model, X_train, y_train, cv=kfold, error_score="raise")
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[54]:


y_pred = logreg_model.predict(X_test)


# In[55]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n", cm)


# In[56]:


from sklearn.metrics import accuracy_score


# In[57]:


accuracy_score(y_test, y_pred)


# In[58]:


group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                cm.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cm.flatten()/np.sum(cm)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm, annot=labels, fmt='', cmap='Reds')


# In[59]:


import statsmodels.api as sm
log_reg_summary = sm.Logit(y_train, X_train).fit()
print(log_reg_summary.summary())


# In[60]:


midseason_corrMatrix = midseason_features.corr()
midseason_features_fig, midseason_features_ax = plt.subplots(figsize=(20,20))
sns.heatmap(midseason_corrMatrix, annot=True, ax=midseason_features_ax)
plt.show()


# In[61]:


midseason_features_copy = midseason_features.copy() 
midseason_features_copy = midseason_features_copy.drop(columns = ["Prev_SRS", "Prev_OeFG%", "Prev_DeFG%"], axis = 1)

midseason_corrMatrix = midseason_features_copy.corr()
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(midseason_corrMatrix, annot=True, ax=ax)


# In[62]:


y = midseason_features_copy['W/L Classify']
X = midseason_features_copy.drop(columns = ['W/L%','Change in W/L%', 'W/L Classify', 'Roster Changes', 'Coach Experience', 'Average Age', 'Young Players', 'Prev_NRtg', 'Prev_OTOV%', 'Prev_ORB%', 'Prev_DFT/FGA%', 'Prev_DTOV%', 'Prev_DRB%'], axis= 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[63]:


X


# In[64]:


kfold = KFold(n_splits=10, random_state=0, shuffle=True)
logreg_model = LogisticRegression(solver='liblinear')
logreg_model.fit(X_train, y_train)
results = cross_val_score(logreg_model, X_train, y_train, cv=kfold, error_score="raise")
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[65]:


y_pred = logreg_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n", cm)


# In[66]:


accuracy_score(y_test, y_pred)


# In[67]:


group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                cm.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cm.flatten()/np.sum(cm)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm, annot=labels, fmt='', cmap='Reds')


# In[68]:


import statsmodels.api as sm
log_reg_summary = sm.Logit(y_train, X_train).fit()
print(log_reg_summary.summary())


# In[69]:


fullseason_features_copy.plot.scatter(x = 'Change in W/L%', y = 'W/L%')
midseason_features_copy.plot.scatter(x = 'Change in W/L%', y = 'W/L%')


# In[70]:


fullseason_features_copy.plot.scatter(x = 'Roster Changes', y = 'W/L%')
midseason_features_copy.plot.scatter(x = 'Roster Changes', y = 'W/L%')


# In[71]:


fullseason_features_copy.plot.scatter(x = 'Coach Experience', y = 'W/L%')
midseason_features_copy.plot.scatter(x = 'Coach Experience', y = 'W/L%')


# In[72]:


fullseason_features_copy.plot.scatter(x = 'Coach Career W/L%', y = 'W/L%')
midseason_features_copy.plot.scatter(x = 'Coach Career W/L%', y = 'W/L%')


# In[73]:


fullseason_features_copy.plot.scatter(x = 'Average Age', y = 'W/L%')
midseason_features_copy.plot.scatter(x = 'Average Age', y = 'W/L%')


# In[74]:


fullseason_features_copy.plot.scatter(x = 'Young Players', y = 'W/L%')
midseason_features_copy.plot.scatter(x = 'Young Players', y = 'W/L%')


# In[75]:


fullseason_features_copy.plot.scatter(x = 'Prev_ORtg', y = 'W/L%')
midseason_features_copy.plot.scatter(x = 'Prev_ORtg', y = 'W/L%')


# In[76]:


fullseason_features_copy.plot.scatter(x = 'Prev_DRtg', y = 'W/L%')
midseason_features_copy.plot.scatter(x = 'Prev_DRtg', y = 'W/L%')


# In[77]:


fullseason_features.plot.scatter(x = 'Prev_SRS', y = 'W/L%')
midseason_features.plot.scatter(x = 'Prev_SRS', y = 'W/L%')


# In[78]:


fullseason_features


# In[79]:


lin_y = fullseason_features['W/L%']
lin_X = (fullseason_features.drop(columns = ['Change in W/L%', 'W/L%', 'Coach Experience', 'Coach Career W/L%', 'Average Age', 'Prev_NRtg', 'Prev_SRS', 'Prev_Pace', 'Prev_OeFG%', 'Prev_OTOV%', 'Prev_ORB%', 'Prev_DeFG%', 'Prev_DTOV%', 'Prev_DRB%', 'Prev_DFT/FGA%', 'W/L Classify'], axis= 1))
lin_X_Coeffs = (fullseason_features.drop(columns = ['Change in W/L%', 'Coach Experience', 'Coach Career W/L%', 'Average Age', 'Prev_NRtg', 'Prev_SRS', 'Prev_Pace', 'Prev_OeFG%', 'Prev_OTOV%', 'Prev_ORB%', 'Prev_DeFG%', 'Prev_DTOV%', 'Prev_DRB%', 'Prev_DFT/FGA%', 'W/L Classify'], axis= 1))
lin_X


# In[80]:


linX_train, linX_test, liny_train, liny_test = train_test_split(lin_X, lin_y, test_size=0.3, random_state=0)
linreg = LinearRegression()

linreg.fit(linX_train, liny_train)

linreg.score(linX_test, liny_test)


# In[81]:


model = sm.OLS(liny_train,linX_train)
results = model.fit()
print(results.summary())


# In[139]:


lin_y = midseason_features['W/L%']
lin_X = (midseason_features.drop(columns = ['W/L%', 'Change in W/L%', 'Young Players', 'Prev_SRS', 'Prev_Pace', 'Prev_OeFG%', 'Prev_OTOV%', 'Prev_ORB%', 'Prev_DeFG%', 'Prev_DTOV%', 'Prev_DRB%', 'W/L Classify'], axis= 1))
lin_X


# In[140]:


linX_train, linX_test, liny_train, liny_test = train_test_split(lin_X, lin_y, test_size=0.3, random_state=0)
linreg = LinearRegression()

linreg.fit(linX_train, liny_train)

linreg.score(linX_test, liny_test)


# In[141]:


model = sm.OLS(liny_train,linX_train)
results = model.fit()
print(results.summary())


# In[85]:


features_copy = features.copy() 
features_copy = features_copy.drop(columns = ['W/L%', 'W/L Classify', 'Midseason Hire', 'Roster Changes', 'Coach Experience', 'Average Age', 'Young Players', 'Prev_ORB%', 'Prev_OFT/FGA%', 'Prev_DTOV%', 'Prev_DRB%', "Prev_OeFG%", "Prev_DeFG%", "Prev_NRtg", "Prev_SRS"], axis = 1)

corrMatrix = features_copy.corr()
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(corrMatrix, annot=True,  annot_kws={"size": 24}, ax=ax)


# In[86]:


fullseason_features_copy = fullseason_features.copy() 
fullseason_features_copy = fullseason_features_copy.drop(columns = ["Prev_NRtg", "Prev_SRS", "Prev_OeFG%", "Prev_DeFG%"], axis = 1)

fullseason_corrMatrix = fullseason_features_copy.corr()
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(fullseason_corrMatrix, annot=True, ax=ax)


# In[87]:


fullseason_features_copy = fullseason_features.copy() 
fullseason_features_copy = fullseason_features_copy.drop(columns = ['W/L%', 'W/L Classify', 'Prev_Pace', 'Roster Changes', 'Coach Experience', 'Average Age', 'Prev_ORB%', 'Prev_OFT/FGA%', 'Prev_DTOV%', 'Prev_DRB%', "Prev_OeFG%", "Prev_DeFG%", "Prev_NRtg", "Prev_SRS"], axis = 1)
#                                                                  ['W/L%','Change in W/L%', 'W/L Classify', 'Roster Changes', 'Coach Experience', 'Average Age', 'Prev_Pace', 'Prev_ORB%', 'Prev_OFT/FGA%', 'Prev_DTOV%', 'Prev_DRB%']
fullseason_corrMatrix = fullseason_features_copy.corr()
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(fullseason_corrMatrix, annot=True, annot_kws={"size": 24}, ax=ax)


# In[88]:


from scipy import stats
stats.ttest_ind(fullseason_hires, midseason_hires, equal_var=True)


# In[89]:


# Define continuous variables

continuous_var = ['Change in W/L%', 'Coach Career W/L%', 'Average Age', 'Prev_ORtg', 'Prev_DRtg', 'Prev_Pace', 'Prev_OTOV%', 'Prev_ORB%', 'Prev_OFT/FGA%', 'Prev_DTOV%', 'Prev_DRB%', 'Prev_DFT/FGA%', 'Prev_W/L%']

# Add logit transform interaction terms (natural log) for continuous variables e.g.. Age * Log(Age)
for var in continuous_var:
    all_coaching_changes[f'{var}:Log_{var}'] = all_coaching_changes[var].apply(lambda x: x * np.log(x))

# Keep columns related to continuous variables
cols_to_keep = continuous_var + all_coaching_changes.columns.tolist()[-len(continuous_var):]

# Redefining variables to include interaction terms
X_lt = all_coaching_changes[cols_to_keep]
X_lt


# In[90]:


import statsmodels.api as sm


# In[91]:


# Add constant term
X_lt = sm.add_constant(X_lt, prepend=False)


# In[92]:


X_lt.dropna(inplace=True)


# In[93]:


y_lt = X_lt.filter(['Change in W/L%'], axis=1)


# In[94]:


X_lt.drop(['Change in W/L%', 'Change in W/L%:Log_Change in W/L%'], axis=1, inplace=True)
X_lt


# In[95]:


print(X_lt.shape, y_lt.shape)


# In[96]:


# Building model and fit the data (using statsmodel's Logit)
logit_results = sm.GLM(y_lt, X_lt, family=sm.families.Binomial()).fit()

# Display summary results
print(logit_results.summary())


# In[97]:


midseason_features_copy = midseason_features.copy() 
midseason_features_copy


# In[98]:


midseason_features_copy = midseason_features_copy.drop(columns = ['W/L%', 'W/L Classify', 'Roster Changes', 'Coach Experience', 'Average Age', 'Young Players', 'Prev_NRtg', 'Prev_OTOV%', 'Prev_ORB%', 'Prev_DFT/FGA%', 'Prev_DTOV%', 'Prev_DRB%', 'Prev_OeFG%', 'Prev_DeFG%'], axis = 1)
#                                                                  ['W/L%', 'W/L Classify', 'Roster Changes', 'Coach Experience', 'Average Age', 'Young Players', 'Prev_NRtg', 'Prev_OTOV%', 'Prev_ORB%', 'Prev_DFT/FGA%', 'Prev_DTOV%', 'Prev_DRB%']


# In[99]:


midseason_features_copy = midseason_features_copy.drop(columns = ['Prev_SRS'], axis = 1)

midseason_corrMatrix = midseason_features_copy.corr()
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(midseason_corrMatrix, annot=True, annot_kws={"size": 24}, ax=ax)


# In[100]:


features.plot.scatter(x = 'Young Players', y = 'W/L%')


# In[101]:


features_copy


# In[102]:


lin_y = features['W/L%']
lin_X = (features_copy.drop(columns = ['Change in W/L%'], axis= 1))
lin_y


# In[103]:


linX_train, linX_test, liny_train, liny_test = train_test_split(lin_X, lin_y, test_size=0.3, random_state=0)

linX_test


# In[104]:


from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

linreg.fit(linX_train, liny_train)

liny_pred = linreg.predict(linX_test)
print(liny_pred)


# In[105]:


linreg.score(linX_test, liny_test)


# In[106]:


residuals = liny_test - liny_pred
residuals


# In[107]:


plt.scatter(liny_pred, residuals)


# In[108]:


normality_plot = sm.qqplot(residuals, line = 'r')


# In[109]:


from statsmodels.stats.diagnostic import het_breuschpagan
bptest = het_breuschpagan(residuals, linX_test)[1]
print('The p value of Breusch-Pagan test is', bptest)


# In[110]:


from statsmodels.stats.stattools import durbin_watson
durbin_watson(residuals)


# In[111]:


features_copy


# In[112]:


lin_y


# In[113]:


lin_X


# In[114]:


lin_X_copy = lin_X.copy()
lin_X_copy


# In[115]:


list(lin_y)


# In[116]:


lin_X_copy.insert(0, "W/L%", list(lin_y), True)


# In[117]:


lin_X_copy


# In[118]:


lin_X_copy = (lin_X_copy.drop(columns = ['Prev_Pace'], axis= 1))

lin_allcoachingchanges_corrMatrix = lin_X_copy.corr()
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(lin_allcoachingchanges_corrMatrix, annot=True, annot_kws={"size": 24}, ax=ax)


# In[119]:


corrMatrix = features_copy.corr()
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(corrMatrix, annot=True, ax=ax)


# In[120]:


features


# In[121]:


features_copy = features.copy() 
features_copy = features_copy.drop(columns = ['Change in W/L%', 'W/L Classify', 'Midseason Hire', 'Coach Experience', 'Young Players', 'Prev_NRtg', 'Prev_SRS', 'Prev_Pace', 'Prev_OeFG%', 'Prev_OTOV%', 'Prev_ORB%', 'Prev_DeFG%', 'Prev_DTOV%', 'Prev_DRB%', 'Prev_DFT/FGA%'], axis = 1)

corrMatrix = features_copy.corr()
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(corrMatrix, annot=True,  annot_kws={"size": 24}, ax=ax)


# In[122]:


fullseason_features


# In[123]:


midseason_features


# In[124]:


fullseason_corrMatrix = fullseason_features.corr()
fullseason_features_fig, fullseason_features_ax = plt.subplots(figsize=(20,20))
sns.heatmap(fullseason_corrMatrix, annot=True, ax=fullseason_features_ax)
plt.show()


# In[125]:


midseason_corrMatrix = midseason_features.corr()
midseason_features_fig, midseason_features_ax = plt.subplots(figsize=(20,20))
sns.heatmap(midseason_corrMatrix, annot=True, ax=midseason_features_ax)
plt.show()


# In[126]:


fullseason_features.plot.scatter(x = 'Prev_OFT/FGA%', y = 'W/L%')
midseason_features.plot.scatter(x = 'Prev_OFT/FGA%', y = 'W/L%')


# In[127]:


fullseason_features.plot.scatter(x = 'Prev_W/L%', y = 'W/L%')
midseason_features.plot.scatter(x = 'Prev_W/L%', y = 'W/L%')


# In[128]:


fullseason_features_copy = fullseason_features.copy() 
fullseason_features_copy = fullseason_features_copy.drop(columns = ['Change in W/L%', 'Coach Experience', 'Coach Career W/L%', 'Average Age', 'Prev_NRtg', 'Prev_SRS', 'Prev_Pace', 'Prev_OeFG%', 'Prev_OTOV%', 'Prev_ORB%', 'Prev_DeFG%', 'Prev_DTOV%', 'Prev_DRB%', 'Prev_DFT/FGA%', 'W/L Classify'], axis = 1)

fullseason_corrMatrix = fullseason_features_copy.corr()
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(fullseason_corrMatrix, annot=True, annot_kws={"size": 24}, ax=ax)


# In[129]:


midseason_features_copy = midseason_features.copy() 
midseason_features_copy = midseason_features_copy.drop(columns = ['Change in W/L%', 'Young Players', 'Prev_SRS', 'Prev_Pace', 'Prev_OeFG%', 'Prev_OTOV%', 'Prev_ORB%', 'Prev_DeFG%', 'Prev_DTOV%', 'Prev_DRB%', 'W/L Classify'], axis = 1)

midseason_corrMatrix = midseason_features_copy.corr()
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(midseason_corrMatrix, annot=True, annot_kws={"size": 24}, ax=ax)


# In[130]:


fullseason_features_copy.plot.scatter(x = 'Roster Changes', y = 'W/L%')
fullseason_features_copy.plot.scatter(x = 'Young Players', y = 'W/L%')
fullseason_features_copy.plot.scatter(x = 'Prev_ORtg', y = 'W/L%')
fullseason_features_copy.plot.scatter(x = 'Prev_DRtg', y = 'W/L%')
fullseason_features_copy.plot.scatter(x = 'Prev_OFT/FGA%', y = 'W/L%')
fullseason_features_copy.plot.scatter(x = 'Prev_W/L%', y = 'W/L%')


# In[131]:


midseason_features_copy.plot.scatter(x = 'Roster Changes', y = 'W/L%', c='orange')
midseason_features_copy.plot.scatter(x = 'Coach Experience', y = 'W/L%', c='orange')
midseason_features_copy.plot.scatter(x = 'Coach Career W/L%', y = 'W/L%', c='orange')


# In[132]:


midseason_features_copy.plot.scatter(x = 'Average Age', y = 'W/L%', c='orange')
midseason_features_copy.plot.scatter(x = 'Prev_ORtg', y = 'W/L%', c='orange')
midseason_features_copy.plot.scatter(x = 'Prev_DRtg', y = 'W/L%', c='orange')


# In[133]:


midseason_features_copy.plot.scatter(x = 'Prev_NRtg', y = 'W/L%', c='orange')
midseason_features_copy.plot.scatter(x = 'Prev_OFT/FGA%', y = 'W/L%', c='orange')
midseason_features_copy.plot.scatter(x = 'Prev_DFT/FGA%', y = 'W/L%', c='orange')
midseason_features_copy.plot.scatter(x = 'Prev_W/L%', y = 'W/L%', c='orange')


# In[134]:


new_coaching_hires = pd.read_csv("Desktop/SAAS/new_coaching_hires.csv")
new_coaching_hires


# In[135]:


fullseason_features


# In[136]:


y = fullseason_features['W/L Classify']
X = fullseason_features.drop(columns = ['W/L%','Change in W/L%', 'W/L Classify', 'Roster Changes', 'Coach Experience', 'Average Age', 'Prev_Pace', 'Prev_ORB%', 'Prev_OFT/FGA%', 'Prev_DTOV%', 'Prev_DRB%', 'Prev_NRtg', 'Prev_SRS', 'Prev_OeFG%', 'Prev_DeFG%'], axis= 1)

X


# In[137]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

kfold = KFold(n_splits=10, random_state=0, shuffle=True)
kfold

logreg_model = LogisticRegression(solver='liblinear')
logreg_model.fit(X_train, y_train)
results = cross_val_score(logreg_model, X_train, y_train, cv=kfold, error_score="raise")
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[ ]:




