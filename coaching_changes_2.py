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


hawks = pd.read_csv("Desktop/SAAS/hawks.csv")

hawks.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

hawks = hawks[(hawks['Coaches'].str.split('(').str[0] != hawks['Coaches'].shift(1).str.split('(').str[0]) | (hawks['Coaches'].str.split('(').str[0] != hawks['Coaches'].shift(-1).str.split('(').str[0])]

hawks = hawks.dropna(axis = 1)

hawks = hawks.reset_index()

hawks["Change in W/L%"] = 0
hawks["New Coach"] = False
hawks["Midseason Hire"] = False
hawks["Prev_W/L%"] = 0

i = 0
while i < len(hawks) - 1:
    if (hawks["Coaches"].loc[i].split("("))[0] != (hawks["Coaches"].loc[i + 1].split("("))[0]:
        hawks["Prev_W/L%"].loc[i] = hawks["W/L%"].loc[i + 1]
        hawks["Change in W/L%"].loc[i] = hawks["W/L%"].loc[i] - hawks["W/L%"].loc[i + 1]
        hawks["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(hawks) - 1:
    if hawks["Season"].loc[k] == hawks["Season"].loc[k + 1]:
        hawks["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(hawks) - 1
while j >= 0:
    if not hawks["New Coach"].loc[j]:
        hawks.drop(index=hawks.index[j], axis=0, inplace=True)
    
    j -= 1

hawks


# In[3]:


celtics = pd.read_csv("Desktop/SAAS/celtics.csv")

celtics.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

celtics = celtics[(celtics['Coaches'].str.split('(').str[0] != celtics['Coaches'].shift(1).str.split('(').str[0]) | (celtics['Coaches'].str.split('(').str[0] != celtics['Coaches'].shift(-1).str.split('(').str[0])]

celtics = celtics.dropna(axis = 1)

celtics = celtics.reset_index()

celtics["Change in W/L%"] = 0
celtics["New Coach"] = False
celtics["Midseason Hire"] = False
celtics["Prev_W/L%"] = 0

i = 0
while i < len(celtics) - 1:
    if (celtics["Coaches"].loc[i].split("("))[0] != (celtics["Coaches"].loc[i + 1].split("("))[0]:
        celtics["Prev_W/L%"].loc[i] = celtics["W/L%"].loc[i + 1]
        celtics["Change in W/L%"].loc[i] = celtics["W/L%"].loc[i] - celtics["W/L%"].loc[i + 1]
        celtics["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(celtics) - 1:
    if celtics["Season"].loc[k] == celtics["Season"].loc[k + 1]:
        celtics["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(celtics) - 1
while j >= 0:
    if not celtics["New Coach"].loc[j]:
        celtics.drop(index=celtics.index[j], axis=0, inplace=True)
    
    j -= 1

celtics


# In[4]:


nets = pd.read_csv("Desktop/SAAS/nets.csv")

nets.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

nets = nets[(nets['Coaches'].str.split('(').str[0] != nets['Coaches'].shift(1).str.split('(').str[0]) | (nets['Coaches'].str.split('(').str[0] != nets['Coaches'].shift(-1).str.split('(').str[0])]

nets = nets.dropna(axis = 1)

nets = nets.reset_index()

nets["Change in W/L%"] = 0
nets["New Coach"] = False
nets["Midseason Hire"] = False
nets["Prev_W/L%"] = 0

i = 0
while i < len(nets) - 1:
    if (nets["Coaches"].loc[i].split("("))[0] != (nets["Coaches"].loc[i + 1].split("("))[0]:
        nets["Prev_W/L%"].loc[i] = nets["W/L%"].loc[i + 1]
        nets["Change in W/L%"].loc[i] = nets["W/L%"].loc[i] - nets["W/L%"].loc[i + 1]
        nets["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(nets) - 1:
    if nets["Season"].loc[k] == nets["Season"].loc[k + 1]:
        nets["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(nets) - 1
while j >= 0:
    if not nets["New Coach"].loc[j]:
        nets.drop(index=nets.index[j], axis=0, inplace=True)
    
    j -= 1

nets


# In[5]:


hornets = pd.read_csv("Desktop/SAAS/hornets.csv")

hornets.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

hornets = hornets[(hornets['Coaches'].str.split('(').str[0] != hornets['Coaches'].shift(1).str.split('(').str[0]) | (hornets['Coaches'].str.split('(').str[0] != hornets['Coaches'].shift(-1).str.split('(').str[0])]

hornets = hornets.dropna(axis = 1)

hornets = hornets.reset_index()

hornets["Change in W/L%"] = 0
hornets["New Coach"] = False
hornets["Midseason Hire"] = False
hornets["Prev_W/L%"] = 0

i = 0
while i < len(hornets) - 1:
    if (hornets["Coaches"].loc[i].split("("))[0] != (hornets["Coaches"].loc[i + 1].split("("))[0]:
        hornets["Prev_W/L%"].loc[i] = hornets["W/L%"].loc[i + 1]
        hornets["Change in W/L%"].loc[i] = hornets["W/L%"].loc[i] - hornets["W/L%"].loc[i + 1]
        hornets["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(hornets) - 1:
    if hornets["Season"].loc[k] == hornets["Season"].loc[k + 1]:
        hornets["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(hornets) - 1
while j >= 0:
    if not hornets["New Coach"].loc[j]:
        hornets.drop(index=hornets.index[j], axis=0, inplace=True)
    
    j -= 1

hornets


# In[6]:


bulls = pd.read_csv("Desktop/SAAS/bulls.csv")

bulls.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

bulls = bulls[(bulls['Coaches'].str.split('(').str[0] != bulls['Coaches'].shift(1).str.split('(').str[0]) | (bulls['Coaches'].str.split('(').str[0] != bulls['Coaches'].shift(-1).str.split('(').str[0])]

bulls = bulls.dropna(axis = 1)

bulls = bulls.reset_index()

bulls["Change in W/L%"] = 0
bulls["New Coach"] = False
bulls["Midseason Hire"] = False
bulls["Prev_W/L%"] = 0

i = 0
while i < len(bulls) - 1:
    if (bulls["Coaches"].loc[i].split("("))[0] != (bulls["Coaches"].loc[i + 1].split("("))[0]:
        bulls["Prev_W/L%"].loc[i] = bulls["W/L%"].loc[i + 1]
        bulls["Change in W/L%"].loc[i] = bulls["W/L%"].loc[i] - bulls["W/L%"].loc[i + 1]
        bulls["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(bulls) - 1:
    if bulls["Season"].loc[k] == bulls["Season"].loc[k + 1]:
        bulls["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(bulls) - 1
while j >= 0:
    if not bulls["New Coach"].loc[j]:
        bulls.drop(index=bulls.index[j], axis=0, inplace=True)
    
    j -= 1

bulls


# In[7]:


cavaliers = pd.read_csv("Desktop/SAAS/cavaliers.csv")

cavaliers.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

cavaliers = cavaliers[(cavaliers['Coaches'].str.split('(').str[0] != cavaliers['Coaches'].shift(1).str.split('(').str[0]) | (cavaliers['Coaches'].str.split('(').str[0] != cavaliers['Coaches'].shift(-1).str.split('(').str[0])]

cavaliers = cavaliers.dropna(axis = 1)

cavaliers = cavaliers.reset_index()

cavaliers["Change in W/L%"] = 0
cavaliers["New Coach"] = False
cavaliers["Midseason Hire"] = False
cavaliers["Prev_W/L%"] = 0

i = 0
while i < len(cavaliers) - 1:
    if (cavaliers["Coaches"].loc[i].split("("))[0] != (cavaliers["Coaches"].loc[i + 1].split("("))[0]:
        cavaliers["Prev_W/L%"].loc[i] = cavaliers["W/L%"].loc[i + 1]
        cavaliers["Change in W/L%"].loc[i] = cavaliers["W/L%"].loc[i] - cavaliers["W/L%"].loc[i + 1]
        cavaliers["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(cavaliers) - 1:
    if cavaliers["Season"].loc[k] == cavaliers["Season"].loc[k + 1]:
        cavaliers["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(cavaliers) - 1
while j >= 0:
    if not cavaliers["New Coach"].loc[j]:
        cavaliers.drop(index=cavaliers.index[j], axis=0, inplace=True)
    
    j -= 1

cavaliers


# In[8]:


mavs = pd.read_csv("Desktop/SAAS/mavs.csv")

mavs.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

mavs = mavs[(mavs['Coaches'].str.split('(').str[0] != mavs['Coaches'].shift(1).str.split('(').str[0]) | (mavs['Coaches'].str.split('(').str[0] != mavs['Coaches'].shift(-1).str.split('(').str[0])]

mavs = mavs.dropna(axis = 1)

mavs = mavs.reset_index()

mavs["Change in W/L%"] = 0
mavs["New Coach"] = False
mavs["Midseason Hire"] = False
mavs["Prev_W/L%"] = 0

i = 0
while i < len(mavs) - 1:
    if (mavs["Coaches"].loc[i].split("("))[0] != (mavs["Coaches"].loc[i + 1].split("("))[0]:
        mavs["Prev_W/L%"].loc[i] = mavs["W/L%"].loc[i + 1]
        mavs["Change in W/L%"].loc[i] = mavs["W/L%"].loc[i] - mavs["W/L%"].loc[i + 1]
        mavs["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(mavs) - 1:
    if mavs["Season"].loc[k] == mavs["Season"].loc[k + 1]:
        mavs["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(mavs) - 1
while j >= 0:
    if not mavs["New Coach"].loc[j]:
        mavs.drop(index=mavs.index[j], axis=0, inplace=True)
    
    j -= 1

mavs


# In[9]:


nuggets = pd.read_csv("Desktop/SAAS/nuggets.csv")

nuggets.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

nuggets = nuggets[(nuggets['Coaches'].str.split('(').str[0] != nuggets['Coaches'].shift(1).str.split('(').str[0]) | (nuggets['Coaches'].str.split('(').str[0] != nuggets['Coaches'].shift(-1).str.split('(').str[0])]

nuggets = nuggets.dropna(axis = 1)

nuggets = nuggets.reset_index()

nuggets["Change in W/L%"] = 0
nuggets["New Coach"] = False
nuggets["Midseason Hire"] = False
nuggets["Prev_W/L%"] = 0

i = 0
while i < len(nuggets) - 1:
    if (nuggets["Coaches"].loc[i].split("("))[0] != (nuggets["Coaches"].loc[i + 1].split("("))[0]:
        nuggets["Prev_W/L%"].loc[i] = nuggets["W/L%"].loc[i + 1]
        nuggets["Change in W/L%"].loc[i] = nuggets["W/L%"].loc[i] - nuggets["W/L%"].loc[i + 1]
        nuggets["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(nuggets) - 1:
    if nuggets["Season"].loc[k] == nuggets["Season"].loc[k + 1]:
        nuggets["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(nuggets) - 1
while j >= 0:
    if not nuggets["New Coach"].loc[j]:
        nuggets.drop(index=nuggets.index[j], axis=0, inplace=True)
    
    j -= 1

nuggets


# In[10]:


pistons = pd.read_csv("Desktop/SAAS/pistons.csv")

pistons.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

pistons = pistons[(pistons['Coaches'].str.split('(').str[0] != pistons['Coaches'].shift(1).str.split('(').str[0]) | (pistons['Coaches'].str.split('(').str[0] != pistons['Coaches'].shift(-1).str.split('(').str[0])]

pistons = pistons.dropna(axis = 1)

pistons = pistons.reset_index()

pistons["Change in W/L%"] = 0
pistons["New Coach"] = False
pistons["Midseason Hire"] = False
pistons["Prev_W/L%"] = 0

i = 0
while i < len(pistons) - 1:
    if (pistons["Coaches"].loc[i].split("("))[0] != (pistons["Coaches"].loc[i + 1].split("("))[0]:
        pistons["Prev_W/L%"].loc[i] = pistons["W/L%"].loc[i + 1]
        pistons["Change in W/L%"].loc[i] = pistons["W/L%"].loc[i] - pistons["W/L%"].loc[i + 1]
        pistons["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(pistons) - 1:
    if pistons["Season"].loc[k] == pistons["Season"].loc[k + 1]:
        pistons["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(pistons) - 1
while j >= 0:
    if not pistons["New Coach"].loc[j]:
        pistons.drop(index=pistons.index[j], axis=0, inplace=True)
    
    j -= 1

pistons


# In[11]:


warriors = pd.read_csv("Desktop/SAAS/warriors.csv")

warriors.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

warriors = warriors[(warriors['Coaches'].str.split('(').str[0] != warriors['Coaches'].shift(1).str.split('(').str[0]) | (warriors['Coaches'].str.split('(').str[0] != warriors['Coaches'].shift(-1).str.split('(').str[0])]

warriors = warriors.dropna(axis = 1)

warriors = warriors.reset_index()

warriors["Change in W/L%"] = 0
warriors["New Coach"] = False
warriors["Midseason Hire"] = False
warriors["Prev_W/L%"] = 0

i = 0
while i < len(warriors) - 1:
    if (warriors["Coaches"].loc[i].split("("))[0] != (warriors["Coaches"].loc[i + 1].split("("))[0]:
        warriors["Prev_W/L%"].loc[i] = warriors["W/L%"].loc[i + 1]
        warriors["Change in W/L%"].loc[i] = warriors["W/L%"].loc[i] - warriors["W/L%"].loc[i + 1]
        warriors["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(warriors) - 1:
    if warriors["Season"].loc[k] == warriors["Season"].loc[k + 1]:
        warriors["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(warriors) - 1
while j >= 0:
    if not warriors["New Coach"].loc[j]:
        warriors.drop(index=warriors.index[j], axis=0, inplace=True)
    
    j -= 1

warriors


# In[12]:


rockets = pd.read_csv("Desktop/SAAS/rockets.csv")

rockets.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

rockets = rockets[(rockets['Coaches'].str.split('(').str[0] != rockets['Coaches'].shift(1).str.split('(').str[0]) | (rockets['Coaches'].str.split('(').str[0] != rockets['Coaches'].shift(-1).str.split('(').str[0])]

rockets = rockets.dropna(axis = 1)

rockets = rockets.reset_index()

rockets["Change in W/L%"] = 0
rockets["New Coach"] = False
rockets["Midseason Hire"] = False
rockets["Prev_W/L%"] = 0

i = 0
while i < len(rockets) - 1:
    if (rockets["Coaches"].loc[i].split("("))[0] != (rockets["Coaches"].loc[i + 1].split("("))[0]:
        rockets["Prev_W/L%"].loc[i] = rockets["W/L%"].loc[i + 1]
        rockets["Change in W/L%"].loc[i] = rockets["W/L%"].loc[i] - rockets["W/L%"].loc[i + 1]
        rockets["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(rockets) - 1:
    if rockets["Season"].loc[k] == rockets["Season"].loc[k + 1]:
        rockets["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(rockets) - 1
while j >= 0:
    if not rockets["New Coach"].loc[j]:
        rockets.drop(index=rockets.index[j], axis=0, inplace=True)
    
    j -= 1

rockets


# In[13]:


pacers = pd.read_csv("Desktop/SAAS/pacers.csv")

pacers.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

pacers = pacers[(pacers['Coaches'].str.split('(').str[0] != pacers['Coaches'].shift(1).str.split('(').str[0]) | (pacers['Coaches'].str.split('(').str[0] != pacers['Coaches'].shift(-1).str.split('(').str[0])]

pacers = pacers.dropna(axis = 1)

pacers = pacers.reset_index()

pacers["Change in W/L%"] = 0
pacers["New Coach"] = False
pacers["Midseason Hire"] = False
pacers["Prev_W/L%"] = 0

i = 0
while i < len(pacers) - 1:
    if (pacers["Coaches"].loc[i].split("("))[0] != (pacers["Coaches"].loc[i + 1].split("("))[0]:
        pacers["Prev_W/L%"].loc[i] = pacers["W/L%"].loc[i + 1]
        pacers["Change in W/L%"].loc[i] = pacers["W/L%"].loc[i] - pacers["W/L%"].loc[i + 1]
        pacers["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(pacers) - 1:
    if pacers["Season"].loc[k] == pacers["Season"].loc[k + 1]:
        pacers["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(pacers) - 1
while j >= 0:
    if not pacers["New Coach"].loc[j]:
        pacers.drop(index=pacers.index[j], axis=0, inplace=True)
    
    j -= 1

pacers


# In[14]:


clippers = pd.read_csv("Desktop/SAAS/clippers.csv")

clippers.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

clippers = clippers[(clippers['Coaches'].str.split('(').str[0] != clippers['Coaches'].shift(1).str.split('(').str[0]) | (clippers['Coaches'].str.split('(').str[0] != clippers['Coaches'].shift(-1).str.split('(').str[0])]

clippers = clippers.dropna(axis = 1)

clippers = clippers.reset_index()

clippers["Change in W/L%"] = 0
clippers["New Coach"] = False
clippers["Midseason Hire"] = False
clippers["Prev_W/L%"] = 0

i = 0
while i < len(clippers) - 1:
    if (clippers["Coaches"].loc[i].split("("))[0] != (clippers["Coaches"].loc[i + 1].split("("))[0]:
        clippers["Prev_W/L%"].loc[i] = clippers["W/L%"].loc[i + 1]
        clippers["Change in W/L%"].loc[i] = clippers["W/L%"].loc[i] - clippers["W/L%"].loc[i + 1]
        clippers["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(clippers) - 1:
    if clippers["Season"].loc[k] == clippers["Season"].loc[k + 1]:
        clippers["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(clippers) - 1
while j >= 0:
    if not clippers["New Coach"].loc[j]:
        clippers.drop(index=clippers.index[j], axis=0, inplace=True)
    
    j -= 1

clippers


# In[15]:


lakers = pd.read_csv("Desktop/SAAS/lakers.csv")

lakers.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

lakers = lakers[(lakers['Coaches'].str.split('(').str[0] != lakers['Coaches'].shift(1).str.split('(').str[0]) | (lakers['Coaches'].str.split('(').str[0] != lakers['Coaches'].shift(-1).str.split('(').str[0])]

lakers = lakers.dropna(axis = 1)

lakers = lakers.reset_index()

lakers["Change in W/L%"] = 0
lakers["New Coach"] = False
lakers["Midseason Hire"] = False
lakers["Prev_W/L%"] = 0

i = 0
while i < len(lakers) - 1:
    if (lakers["Coaches"].loc[i].split("("))[0] != (lakers["Coaches"].loc[i + 1].split("("))[0]:
        lakers["Prev_W/L%"].loc[i] = lakers["W/L%"].loc[i + 1]
        lakers["Change in W/L%"].loc[i] = lakers["W/L%"].loc[i] - lakers["W/L%"].loc[i + 1]
        lakers["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(lakers) - 1:
    if lakers["Season"].loc[k] == lakers["Season"].loc[k + 1]:
        lakers["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(lakers) - 1
while j >= 0:
    if not lakers["New Coach"].loc[j]:
        lakers.drop(index=lakers.index[j], axis=0, inplace=True)
    
    j -= 1

lakers


# In[16]:


grizzlies = pd.read_csv("Desktop/SAAS/grizzlies.csv")

grizzlies.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

grizzlies = grizzlies[(grizzlies['Coaches'].str.split('(').str[0] != grizzlies['Coaches'].shift(1).str.split('(').str[0]) | (grizzlies['Coaches'].str.split('(').str[0] != grizzlies['Coaches'].shift(-1).str.split('(').str[0])]

grizzlies = grizzlies.dropna(axis = 1)

grizzlies = grizzlies.reset_index()

grizzlies["Change in W/L%"] = 0
grizzlies["New Coach"] = False
grizzlies["Midseason Hire"] = False
grizzlies["Prev_W/L%"] = 0

i = 0
while i < len(grizzlies) - 1:
    if (grizzlies["Coaches"].loc[i].split("("))[0] != (grizzlies["Coaches"].loc[i + 1].split("("))[0]:
        grizzlies["Prev_W/L%"].loc[i] = grizzlies["W/L%"].loc[i + 1]
        grizzlies["Change in W/L%"].loc[i] = grizzlies["W/L%"].loc[i] - grizzlies["W/L%"].loc[i + 1]
        grizzlies["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(grizzlies) - 1:
    if grizzlies["Season"].loc[k] == grizzlies["Season"].loc[k + 1]:
        grizzlies["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(grizzlies) - 1
while j >= 0:
    if not grizzlies["New Coach"].loc[j]:
        grizzlies.drop(index=grizzlies.index[j], axis=0, inplace=True)
    
    j -= 1

grizzlies


# In[17]:


heat = pd.read_csv("Desktop/SAAS/heat.csv")

heat.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

heat = heat[(heat['Coaches'].str.split('(').str[0] != heat['Coaches'].shift(1).str.split('(').str[0]) | (heat['Coaches'].str.split('(').str[0] != heat['Coaches'].shift(-1).str.split('(').str[0])]

heat = heat.dropna(axis = 1)

heat = heat.reset_index()

heat["Change in W/L%"] = 0
heat["New Coach"] = False
heat["Midseason Hire"] = False
heat["Prev_W/L%"] = 0

i = 0
while i < len(heat) - 1:
    if (heat["Coaches"].loc[i].split("("))[0] != (heat["Coaches"].loc[i + 1].split("("))[0]:
        heat["Prev_W/L%"].loc[i] = heat["W/L%"].loc[i + 1]
        heat["Change in W/L%"].loc[i] = heat["W/L%"].loc[i] - heat["W/L%"].loc[i + 1]
        heat["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(heat) - 1:
    if heat["Season"].loc[k] == heat["Season"].loc[k + 1]:
        heat["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(heat) - 1
while j >= 0:
    if not heat["New Coach"].loc[j]:
        heat.drop(index=heat.index[j], axis=0, inplace=True)
    
    j -= 1

heat


# In[18]:


bucks = pd.read_csv("Desktop/SAAS/bucks.csv")

bucks.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

bucks = bucks[(bucks['Coaches'].str.split('(').str[0] != bucks['Coaches'].shift(1).str.split('(').str[0]) | (bucks['Coaches'].str.split('(').str[0] != bucks['Coaches'].shift(-1).str.split('(').str[0])]

bucks = bucks.dropna(axis = 1)

bucks = bucks.reset_index()

bucks["Change in W/L%"] = 0
bucks["New Coach"] = False
bucks["Midseason Hire"] = False
bucks["Prev_W/L%"] = 0

i = 0
while i < len(bucks) - 1:
    if (bucks["Coaches"].loc[i].split("("))[0] != (bucks["Coaches"].loc[i + 1].split("("))[0]:
        bucks["Prev_W/L%"].loc[i] = bucks["W/L%"].loc[i + 1]
        bucks["Change in W/L%"].loc[i] = bucks["W/L%"].loc[i] - bucks["W/L%"].loc[i + 1]
        bucks["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(bucks) - 1:
    if bucks["Season"].loc[k] == bucks["Season"].loc[k + 1]:
        bucks["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(bucks) - 1
while j >= 0:
    if not bucks["New Coach"].loc[j]:
        bucks.drop(index=bucks.index[j], axis=0, inplace=True)
    
    j -= 1

bucks


# In[19]:


twolves = pd.read_csv("Desktop/SAAS/twolves.csv")

twolves.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

twolves = twolves[(twolves['Coaches'].str.split('(').str[0] != twolves['Coaches'].shift(1).str.split('(').str[0]) | (twolves['Coaches'].str.split('(').str[0] != twolves['Coaches'].shift(-1).str.split('(').str[0])]

twolves = twolves.dropna(axis = 1)

twolves = twolves.reset_index()

twolves["Change in W/L%"] = 0
twolves["New Coach"] = False
twolves["Midseason Hire"] = False
twolves["Prev_W/L%"] = 0

i = 0
while i < len(twolves) - 1:
    if (twolves["Coaches"].loc[i].split("("))[0] != (twolves["Coaches"].loc[i + 1].split("("))[0]:
        twolves["Prev_W/L%"].loc[i] = twolves["W/L%"].loc[i + 1]
        twolves["Change in W/L%"].loc[i] = twolves["W/L%"].loc[i] - twolves["W/L%"].loc[i + 1]
        twolves["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(twolves) - 1:
    if twolves["Season"].loc[k] == twolves["Season"].loc[k + 1]:
        twolves["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(twolves) - 1
while j >= 0:
    if not twolves["New Coach"].loc[j]:
        twolves.drop(index=twolves.index[j], axis=0, inplace=True)
    
    j -= 1

twolves


# In[20]:


pelicans = pd.read_csv("Desktop/SAAS/pelicans.csv")

pelicans.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

pelicans = pelicans[(pelicans['Coaches'].str.split('(').str[0] != pelicans['Coaches'].shift(1).str.split('(').str[0]) | (pelicans['Coaches'].str.split('(').str[0] != pelicans['Coaches'].shift(-1).str.split('(').str[0])]

pelicans = pelicans.dropna(axis = 1)

pelicans = pelicans.reset_index()

pelicans["Change in W/L%"] = 0
pelicans["New Coach"] = False
pelicans["Midseason Hire"] = False
pelicans["Prev_W/L%"] = 0

i = 0
while i < len(pelicans) - 1:
    if (pelicans["Coaches"].loc[i].split("("))[0] != (pelicans["Coaches"].loc[i + 1].split("("))[0]:
        pelicans["Prev_W/L%"].loc[i] = pelicans["W/L%"].loc[i + 1]
        pelicans["Change in W/L%"].loc[i] = pelicans["W/L%"].loc[i] - pelicans["W/L%"].loc[i + 1]
        pelicans["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(pelicans) - 1:
    if pelicans["Season"].loc[k] == pelicans["Season"].loc[k + 1]:
        pelicans["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(pelicans) - 1
while j >= 0:
    if not pelicans["New Coach"].loc[j]:
        pelicans.drop(index=pelicans.index[j], axis=0, inplace=True)
    
    j -= 1

pelicans


# In[21]:


knicks = pd.read_csv("Desktop/SAAS/knicks.csv")

knicks.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

knicks = knicks[(knicks['Coaches'].str.split('(').str[0] != knicks['Coaches'].shift(1).str.split('(').str[0]) | (knicks['Coaches'].str.split('(').str[0] != knicks['Coaches'].shift(-1).str.split('(').str[0])]

knicks = knicks.dropna(axis = 1)

knicks = knicks.reset_index()

knicks["Change in W/L%"] = 0
knicks["New Coach"] = False
knicks["Midseason Hire"] = False
knicks["Prev_W/L%"] = 0

i = 0
while i < len(knicks) - 1:
    if (knicks["Coaches"].loc[i].split("("))[0] != (knicks["Coaches"].loc[i + 1].split("("))[0]:
        knicks["Prev_W/L%"].loc[i] = knicks["W/L%"].loc[i + 1]
        knicks["Change in W/L%"].loc[i] = knicks["W/L%"].loc[i] - knicks["W/L%"].loc[i + 1]
        knicks["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(knicks) - 1:
    if knicks["Season"].loc[k] == knicks["Season"].loc[k + 1]:
        knicks["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(knicks) - 1
while j >= 0:
    if not knicks["New Coach"].loc[j]:
        knicks.drop(index=knicks.index[j], axis=0, inplace=True)
    
    j -= 1

knicks


# In[22]:


thunder = pd.read_csv("Desktop/SAAS/thunder.csv")

thunder.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

thunder = thunder[(thunder['Coaches'].str.split('(').str[0] != thunder['Coaches'].shift(1).str.split('(').str[0]) | (thunder['Coaches'].str.split('(').str[0] != thunder['Coaches'].shift(-1).str.split('(').str[0])]

thunder = thunder.dropna(axis = 1)

thunder = thunder.reset_index()

thunder["Change in W/L%"] = 0
thunder["New Coach"] = False
thunder["Midseason Hire"] = False
thunder["Prev_W/L%"] = 0

i = 0
while i < len(thunder) - 1:
    if (thunder["Coaches"].loc[i].split("("))[0] != (thunder["Coaches"].loc[i + 1].split("("))[0]:
        thunder["Prev_W/L%"].loc[i] = thunder["W/L%"].loc[i + 1]
        thunder["Change in W/L%"].loc[i] = thunder["W/L%"].loc[i] - thunder["W/L%"].loc[i + 1]
        thunder["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(thunder) - 1:
    if thunder["Season"].loc[k] == thunder["Season"].loc[k + 1]:
        thunder["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(thunder) - 1
while j >= 0:
    if not thunder["New Coach"].loc[j]:
        thunder.drop(index=thunder.index[j], axis=0, inplace=True)
    
    j -= 1

thunder


# In[23]:


magic = pd.read_csv("Desktop/SAAS/magic.csv")

magic.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

magic = magic[(magic['Coaches'].str.split('(').str[0] != magic['Coaches'].shift(1).str.split('(').str[0]) | (magic['Coaches'].str.split('(').str[0] != magic['Coaches'].shift(-1).str.split('(').str[0])]

magic = magic.dropna(axis = 1)

magic = magic.reset_index()

magic["Change in W/L%"] = 0
magic["New Coach"] = False
magic["Midseason Hire"] = False
magic["Prev_W/L%"] = 0

i = 0
while i < len(magic) - 1:
    if (magic["Coaches"].loc[i].split("("))[0] != (magic["Coaches"].loc[i + 1].split("("))[0]:
        magic["Prev_W/L%"].loc[i] = magic["W/L%"].loc[i + 1]
        magic["Change in W/L%"].loc[i] = magic["W/L%"].loc[i] - magic["W/L%"].loc[i + 1]
        magic["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(magic) - 1:
    if magic["Season"].loc[k] == magic["Season"].loc[k + 1]:
        magic["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(magic) - 1
while j >= 0:
    if not magic["New Coach"].loc[j]:
        magic.drop(index=magic.index[j], axis=0, inplace=True)
    
    j -= 1

magic


# In[24]:


sixers = pd.read_csv("Desktop/SAAS/sixers.csv")

sixers.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

sixers = sixers[(sixers['Coaches'].str.split('(').str[0] != sixers['Coaches'].shift(1).str.split('(').str[0]) | (sixers['Coaches'].str.split('(').str[0] != sixers['Coaches'].shift(-1).str.split('(').str[0])]

sixers = sixers.dropna(axis = 1)

sixers = sixers.reset_index()

sixers["Change in W/L%"] = 0
sixers["New Coach"] = False
sixers["Midseason Hire"] = False
sixers["Prev_W/L%"] = 0

i = 0
while i < len(sixers) - 1:
    if (sixers["Coaches"].loc[i].split("("))[0] != (sixers["Coaches"].loc[i + 1].split("("))[0]:
        sixers["Prev_W/L%"].loc[i] = sixers["W/L%"].loc[i + 1]
        sixers["Change in W/L%"].loc[i] = sixers["W/L%"].loc[i] - sixers["W/L%"].loc[i + 1]
        sixers["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(sixers) - 1:
    if sixers["Season"].loc[k] == sixers["Season"].loc[k + 1]:
        sixers["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(sixers) - 1
while j >= 0:
    if not sixers["New Coach"].loc[j]:
        sixers.drop(index=sixers.index[j], axis=0, inplace=True)
    
    j -= 1

sixers


# In[25]:


suns = pd.read_csv("Desktop/SAAS/suns.csv")

suns.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

suns = suns[(suns['Coaches'].str.split('(').str[0] != suns['Coaches'].shift(1).str.split('(').str[0]) | (suns['Coaches'].str.split('(').str[0] != suns['Coaches'].shift(-1).str.split('(').str[0])]

suns = suns.dropna(axis = 1)

suns = suns.reset_index()

suns["Change in W/L%"] = 0
suns["New Coach"] = False
suns["Midseason Hire"] = False
suns["Prev_W/L%"] = 0

i = 0
while i < len(suns) - 1:
    if (suns["Coaches"].loc[i].split("("))[0] != (suns["Coaches"].loc[i + 1].split("("))[0]:
        suns["Prev_W/L%"].loc[i] = suns["W/L%"].loc[i + 1]
        suns["Change in W/L%"].loc[i] = suns["W/L%"].loc[i] - suns["W/L%"].loc[i + 1]
        suns["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(suns) - 1:
    if suns["Season"].loc[k] == suns["Season"].loc[k + 1]:
        suns["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(suns) - 1
while j >= 0:
    if not suns["New Coach"].loc[j]:
        suns.drop(index=suns.index[j], axis=0, inplace=True)
    
    j -= 1

suns


# In[26]:


trailblazers = pd.read_csv("Desktop/SAAS/trailblazers.csv")

trailblazers.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

trailblazers = trailblazers[(trailblazers['Coaches'].str.split('(').str[0] != trailblazers['Coaches'].shift(1).str.split('(').str[0]) | (trailblazers['Coaches'].str.split('(').str[0] != trailblazers['Coaches'].shift(-1).str.split('(').str[0])]

trailblazers = trailblazers.dropna(axis = 1)

trailblazers = trailblazers.reset_index()

trailblazers["Change in W/L%"] = 0
trailblazers["New Coach"] = False
trailblazers["Midseason Hire"] = False
trailblazers["Prev_W/L%"] = 0

i = 0
while i < len(trailblazers) - 1:
    if (trailblazers["Coaches"].loc[i].split("("))[0] != (trailblazers["Coaches"].loc[i + 1].split("("))[0]:
        trailblazers["Prev_W/L%"].loc[i] = trailblazers["W/L%"].loc[i + 1]
        trailblazers["Change in W/L%"].loc[i] = trailblazers["W/L%"].loc[i] - trailblazers["W/L%"].loc[i + 1]
        trailblazers["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(trailblazers) - 1:
    if trailblazers["Season"].loc[k] == trailblazers["Season"].loc[k + 1]:
        trailblazers["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(trailblazers) - 1
while j >= 0:
    if not trailblazers["New Coach"].loc[j]:
        trailblazers.drop(index=trailblazers.index[j], axis=0, inplace=True)
    
    j -= 1

trailblazers


# In[27]:


kings = pd.read_csv("Desktop/SAAS/kings.csv")

kings.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

kings = kings[(kings['Coaches'].str.split('(').str[0] != kings['Coaches'].shift(1).str.split('(').str[0]) | (kings['Coaches'].str.split('(').str[0] != kings['Coaches'].shift(-1).str.split('(').str[0])]

kings = kings.dropna(axis = 1)

kings = kings.reset_index()

kings["Change in W/L%"] = 0
kings["New Coach"] = False
kings["Midseason Hire"] = False
kings["Prev_W/L%"] = 0

i = 0
while i < len(kings) - 1:
    if (kings["Coaches"].loc[i].split("("))[0] != (kings["Coaches"].loc[i + 1].split("("))[0]:
        kings["Prev_W/L%"].loc[i] = kings["W/L%"].loc[i + 1]
        kings["Change in W/L%"].loc[i] = kings["W/L%"].loc[i] - kings["W/L%"].loc[i + 1]
        kings["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(kings) - 1:
    if kings["Season"].loc[k] == kings["Season"].loc[k + 1]:
        kings["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(kings) - 1
while j >= 0:
    if not kings["New Coach"].loc[j]:
        kings.drop(index=kings.index[j], axis=0, inplace=True)
    
    j -= 1

kings


# In[28]:


spurs = pd.read_csv("Desktop/SAAS/spurs.csv")

spurs.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

spurs = spurs[(spurs['Coaches'].str.split('(').str[0] != spurs['Coaches'].shift(1).str.split('(').str[0]) | (spurs['Coaches'].str.split('(').str[0] != spurs['Coaches'].shift(-1).str.split('(').str[0])]

spurs = spurs.dropna(axis = 1)

spurs = spurs.reset_index()

spurs["Change in W/L%"] = 0
spurs["New Coach"] = False
spurs["Midseason Hire"] = False
spurs["Prev_W/L%"] = 0

i = 0
while i < len(spurs) - 1:
    if (spurs["Coaches"].loc[i].split("("))[0] != (spurs["Coaches"].loc[i + 1].split("("))[0]:
        spurs["Prev_W/L%"].loc[i] = spurs["W/L%"].loc[i + 1]
        spurs["Change in W/L%"].loc[i] = spurs["W/L%"].loc[i] - spurs["W/L%"].loc[i + 1]
        spurs["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(spurs) - 1:
    if spurs["Season"].loc[k] == spurs["Season"].loc[k + 1]:
        spurs["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(spurs) - 1
while j >= 0:
    if not spurs["New Coach"].loc[j]:
        spurs.drop(index=spurs.index[j], axis=0, inplace=True)
    
    j -= 1

spurs


# In[29]:


raptors = pd.read_csv("Desktop/SAAS/raptors.csv")

raptors.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

raptors = raptors[(raptors['Coaches'].str.split('(').str[0] != raptors['Coaches'].shift(1).str.split('(').str[0]) | (raptors['Coaches'].str.split('(').str[0] != raptors['Coaches'].shift(-1).str.split('(').str[0])]

raptors = raptors.dropna(axis = 1)

raptors = raptors.reset_index()

raptors["Change in W/L%"] = 0
raptors["New Coach"] = False
raptors["Midseason Hire"] = False
raptors["Prev_W/L%"] = 0

i = 0
while i < len(raptors) - 1:
    if (raptors["Coaches"].loc[i].split("("))[0] != (raptors["Coaches"].loc[i + 1].split("("))[0]:
        raptors["Prev_W/L%"].loc[i] = raptors["W/L%"].loc[i + 1]
        raptors["Change in W/L%"].loc[i] = raptors["W/L%"].loc[i] - raptors["W/L%"].loc[i + 1]
        raptors["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(raptors) - 1:
    if raptors["Season"].loc[k] == raptors["Season"].loc[k + 1]:
        raptors["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(raptors) - 1
while j >= 0:
    if not raptors["New Coach"].loc[j]:
        raptors.drop(index=raptors.index[j], axis=0, inplace=True)
    
    j -= 1

raptors


# In[30]:


jazz = pd.read_csv("Desktop/SAAS/jazz.csv")

jazz.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

jazz = jazz[(jazz['Coaches'].str.split('(').str[0] != jazz['Coaches'].shift(1).str.split('(').str[0]) | (jazz['Coaches'].str.split('(').str[0] != jazz['Coaches'].shift(-1).str.split('(').str[0])]

jazz = jazz.dropna(axis = 1)

jazz = jazz.reset_index()

jazz["Change in W/L%"] = 0
jazz["New Coach"] = False
jazz["Midseason Hire"] = False
jazz["Prev_W/L%"] = 0

i = 0
while i < len(jazz) - 1:
    if (jazz["Coaches"].loc[i].split("("))[0] != (jazz["Coaches"].loc[i + 1].split("("))[0]:
        jazz["Prev_W/L%"].loc[i] = jazz["W/L%"].loc[i + 1]
        jazz["Change in W/L%"].loc[i] = jazz["W/L%"].loc[i] - jazz["W/L%"].loc[i + 1]
        jazz["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(jazz) - 1:
    if jazz["Season"].loc[k] == jazz["Season"].loc[k + 1]:
        jazz["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(jazz) - 1
while j >= 0:
    if not jazz["New Coach"].loc[j]:
        jazz.drop(index=jazz.index[j], axis=0, inplace=True)
    
    j -= 1

jazz


# In[31]:


wizards = pd.read_csv("Desktop/SAAS/wizards.csv")

wizards.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

wizards = wizards[(wizards['Coaches'].str.split('(').str[0] != wizards['Coaches'].shift(1).str.split('(').str[0]) | (wizards['Coaches'].str.split('(').str[0] != wizards['Coaches'].shift(-1).str.split('(').str[0])]

wizards = wizards.dropna(axis = 1)

wizards = wizards.reset_index()

wizards["Change in W/L%"] = 0
wizards["New Coach"] = False
wizards["Midseason Hire"] = False
wizards["Prev_W/L%"] = 0

i = 0
while i < len(wizards) - 1:
    if (wizards["Coaches"].loc[i].split("("))[0] != (wizards["Coaches"].loc[i + 1].split("("))[0]:
        wizards["Prev_W/L%"].loc[i] = wizards["W/L%"].loc[i + 1]
        wizards["Change in W/L%"].loc[i] = wizards["W/L%"].loc[i] - wizards["W/L%"].loc[i + 1]
        wizards["New Coach"].loc[i] = True
        i = i + 1
    else:
        i = i + 1
        
k = 0
while k < len(wizards) - 1:
    if wizards["Season"].loc[k] == wizards["Season"].loc[k + 1]:
        wizards["Midseason Hire"].loc[k] = True
    k = k + 1
        
        
j = len(wizards) - 1
while j >= 0:
    if not wizards["New Coach"].loc[j]:
        wizards.drop(index=wizards.index[j], axis=0, inplace=True)
    
    j -= 1

wizards


# In[32]:


all_coaching_changes = pd.concat([hawks, celtics, nets, hornets, bulls, cavaliers, mavs, pistons, nuggets, warriors, rockets, pacers, clippers, lakers, grizzlies, heat, bucks, twolves, pelicans, knicks, thunder, magic, sixers, suns, trailblazers, kings, spurs, raptors, jazz, wizards])
all_coaching_changes


# In[33]:


all_coaching_changes = all_coaching_changes.reset_index()
all_coaching_changes


# In[34]:


all_coaching_changes.drop(['level_0', 'index', 'New Coach'], axis = 1, inplace = True)
all_coaching_changes


# In[35]:


pd.set_option("display.max_rows", None, "display.max_columns", None)
print(all_coaching_changes)


# In[36]:


all_coaching_changes


# In[ ]:





# In[93]:


hawks


# In[94]:


hawks["Roster Changes"] = 0
hawks


# In[95]:


hawks["Roster Changes"].loc[2] = 30
hawks


# In[96]:


hawks["Roster Changes"].loc[4] = 22
hawks


# In[97]:


hawks["Roster Changes"].loc[0] = 1
hawks


# In[98]:


hawks["Roster Changes"].loc[10] = 7
hawks


# In[99]:


hawks["Roster Changes"].loc[6] = 11
hawks


# In[100]:


hawks["Roster Changes"].loc[8] = 39
hawks


# In[101]:


hawks["Roster Changes"].loc[12] = 17
hawks


# In[102]:


hawks["Roster Changes"].loc[14] = 16
hawks


# In[103]:


hawks["Roster Changes"].loc[16] = 16
hawks


# In[104]:


hawks["Roster Changes"].loc[18] = 16
hawks


# In[105]:


hawks["Roster Changes"].loc[20] = 9
hawks


# In[107]:


hawks["Coach Experience"] = 0
hawks["Coach Career W/L%"] = 0
hawks["Star Players"] = 0
hawks["Young Players"] = 0
hawks["Prev_ORtg"] = 0
hawks["Prev_DRtg"] = 0
hawks["Prev_NRtg"] = 0
hawks["Prev_SRS"] = 0
hawks["Prev_Pace"] = 0
hawks["Prev_OeFG%"] = 0
hawks["Prev_OTOV%"] = 0
hawks["Prev_ORB%"] = 0
hawks["Prev_OFT/FGA%"] = 0
hawks["Prev_DeFG%"] = 0
hawks["Prev_DTOV%"] = 0
hawks["Prev_DRB%"] = 0
hawks["Prev_DFT/FGA%"] = 0


hawks


# In[108]:


hawks["Coach Experience"].loc[0] = 20
hawks["Coach Experience"].loc[2] = 9
hawks["Coach Experience"].loc[4] = 17
hawks["Coach Experience"].loc[6] = 17
hawks["Coach Experience"].loc[8] = 7
hawks["Coach Experience"].loc[10] = 8
hawks["Coach Experience"].loc[12] = 0
hawks["Coach Experience"].loc[14] = 20
hawks["Coach Experience"].loc[16] = 11
hawks["Coach Experience"].loc[18] = 4
hawks["Coach Experience"].loc[20] = 9


# In[109]:


hawks


# In[110]:


hawks["Coach Career W/L%"].loc[0] = 0.529
hawks["Coach Career W/L%"].loc[2] = 0.000
hawks["Coach Career W/L%"].loc[4] = 0.000
hawks["Coach Career W/L%"].loc[6] = 0.000
hawks["Coach Career W/L%"].loc[8] = 0.000
hawks["Coach Career W/L%"].loc[10] = 0.000
hawks["Coach Career W/L%"].loc[12] = 0.000
hawks["Coach Career W/L%"].loc[14] = 0.537
hawks["Coach Career W/L%"].loc[16] = 0.360
hawks["Coach Career W/L%"].loc[18] = 0.000
hawks["Coach Career W/L%"].loc[20] = 0.467
hawks


# In[111]:


hawks["Star Players"].loc[0] = 1
hawks["Star Players"].loc[2] = 0
hawks["Star Players"].loc[4] = 1
hawks["Star Players"].loc[6] = 2
hawks["Star Players"].loc[8] = 1
hawks["Star Players"].loc[10] = 2
hawks["Star Players"].loc[12] = 1
hawks["Star Players"].loc[14] = 4
hawks["Star Players"].loc[16] = 3
hawks["Star Players"].loc[18] = 2
hawks["Star Players"].loc[20] = 3

hawks["Young Players"].loc[0] = 6
hawks["Young Players"].loc[2] = 9
hawks["Young Players"].loc[4] = 6
hawks["Young Players"].loc[6] = 3
hawks["Young Players"].loc[8] = 6
hawks["Young Players"].loc[10] = 3
hawks["Young Players"].loc[12] = 9
hawks["Young Players"].loc[14] = 2
hawks["Young Players"].loc[16] = 4
hawks["Young Players"].loc[18] = 5
hawks["Young Players"].loc[20] = 5

hawks


# In[112]:


all_coaching_changes


# In[ ]:




