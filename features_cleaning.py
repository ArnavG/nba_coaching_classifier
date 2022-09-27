#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings('ignore')


# In[2]:


all_coaching_changes = pd.read_csv("Desktop/SAAS/final_coaching_changes_1.csv")


# In[3]:


all_coaching_changes


# In[4]:


for i in range(len(all_coaching_changes)):
    year = all_coaching_changes["Season"].loc[i].split("-")[0]
    team = all_coaching_changes["Team"].loc[i]
    
    url = f"https://www.basketball-reference.com/teams/{team}/{year}.html"
    df_list = pd.read_html(url)
    df = df_list[0]
    
    url1 = f"https://www.basketball-reference.com/teams/{team}/{int(year) + 1}.html"
    df_list1 = pd.read_html(url1)
    df1 = df_list1[0]
    
    df = df.append(df1)
    df = df.drop_duplicates(subset=['Player'], keep=False)
    df = df.reset_index()
    
    number_coaching_changes = len(df.index)
    all_coaching_changes["Roster Changes"].loc[i] = number_coaching_changes
    
all_coaching_changes


# In[5]:


pd.set_option("display.max_rows", None, "display.max_columns", None)
all_coaching_changes


# In[6]:


all_coaching_changes.to_clipboard()


# In[7]:


year


# In[11]:


url2 = f"https://www.basketball-reference.com/teams/ATL/2021.html"
df_list2 = pd.read_html(url2)
df2 = df_list2[0]
df2.loc[0]


# In[14]:


i = len(df2) - 1
df3 = []
while i >= 0:
    if df2["Exp"].loc[i] == "R" or df2["Exp"].loc[i] == "1" and int(df2["Birth Date"].loc[i].split(" ")[2]) + 25 >= 2020:
        df3.append(df2.loc[i])
    i -= 1

df3


# In[15]:





# In[16]:


df4


# In[20]:


all_coaching_changes


# In[34]:


df3 = []
prev_num_yp = 0

for i in range(len(all_coaching_changes)):
    
    year = all_coaching_changes["Season"].loc[i].split("-")[0]
    team = all_coaching_changes["Team"].loc[i]

    url2 = f"https://www.basketball-reference.com/teams/{team}/{int(year) + 1}.html"
    df_list2 = pd.read_html(url2)
    df2 = df_list2[0]
    df2.loc[0]
    
    for j in range(len(df2)):
        if df2["Exp"].loc[j] == "R" or df2["Exp"].loc[j] == "1" and int(df2["Birth Date"].loc[j].split(" ")[2]) + 25 >= int(year):
            df3.append(df2.loc[j])
    
    number_young_players = len(df3) - prev_num_yp
    all_coaching_changes["Young Players"].loc[i] = number_young_players
    prev_num_yp = len(df3)

df4 = pd.DataFrame(df3)
df4


# In[35]:


len(df4)


# In[36]:


len(df3)


# In[37]:


all_coaching_changes


# In[ ]:




