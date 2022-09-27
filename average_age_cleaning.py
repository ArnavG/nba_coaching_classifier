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
pd.set_option("display.max_rows", None, "display.max_columns", None)
all_coaching_changes


# In[3]:


for i in range(len(all_coaching_changes)):
    year = int(all_coaching_changes["Season"].loc[i].split("-")[0]) + 1
    team = all_coaching_changes["Team"].loc[i]
    age_sum = 0
    
    url = f"https://www.basketball-reference.com/teams/{team}/{year}.html"
    
    df_list = pd.read_html(url)
    df = df_list[0]

    for j in range(len(df)):
        age_sum += int(year) - int(df["Birth Date"].loc[j].split(" ")[2])

    age_sum /= len(df)

    all_coaching_changes["Average Age"].loc[i] = age_sum
    
all_coaching_changes


# In[4]:


all_coaching_changes.to_clipboard()


# In[ ]:




