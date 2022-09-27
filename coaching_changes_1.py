#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[11]:


hawks = pd.read_csv("Desktop/SAAS/hawks.csv")


# In[12]:


hawks.head()


# In[13]:


hawks.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)


# In[14]:


hawks.head()


# In[15]:


hawks = hawks[(hawks['Coaches'].str.split('(').str[0] != hawks['Coaches'].shift(1).str.split('(').str[0]) | (hawks['Coaches'].str.split('(').str[0] != hawks['Coaches'].shift(-1).str.split('(').str[0])]


# In[16]:


hawks = hawks.dropna(axis = 1)
hawks.drop(index=hawks.index[-1], 
        axis=0, 
        inplace=True)


# In[17]:


hawks


# In[18]:


hawks["Change in W/L%"] = 0


# In[19]:


hawks = hawks.reset_index()


# In[20]:


i = 0
while i < len(hawks) - 1:
    if (hawks["Coaches"].loc[i].split("("))[0] != (hawks["Coaches"].loc[i + 1].split("("))[0]:
        hawks["Change in W/L%"].loc[i] = hawks["W/L%"].loc[i] - hawks["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[21]:


hawks


# In[22]:


celtics = pd.read_csv("Desktop/SAAS/celtics.csv")


# In[23]:


celtics.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)


# In[24]:


celtics = celtics[(celtics['Coaches'].str.split('(').str[0] != celtics['Coaches'].shift(1).str.split('(').str[0]) | (celtics['Coaches'].str.split('(').str[0] != celtics['Coaches'].shift(-1).str.split('(').str[0])]


# In[25]:


celtics = celtics.dropna(axis = 1)


# In[26]:


celtics


# In[27]:


celtics.drop(index=celtics.index[0], 
        axis=0, 
        inplace=True)


# In[28]:


celtics


# In[29]:


celtics = celtics.reset_index()


# In[30]:


celtics["Change in W/L%"] = 0


# In[31]:


celtics


# In[32]:


i = 0
while i < len(celtics) - 1:
    if (celtics["Coaches"].loc[i].split("("))[0] != (celtics["Coaches"].loc[i + 1].split("("))[0]:
        celtics["Change in W/L%"].loc[i] = celtics["W/L%"].loc[i] - celtics["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[33]:


celtics


# In[34]:


nets = pd.read_csv("Desktop/SAAS/nets.csv")


# In[35]:


nets


# In[36]:


nets.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)


# In[37]:


nets


# In[38]:


nets = nets[(nets['Coaches'].str.split('(').str[0] != nets['Coaches'].shift(1).str.split('(').str[0]) | (nets['Coaches'].str.split('(').str[0] != nets['Coaches'].shift(-1).str.split('(').str[0])]


# In[39]:


nets = nets.dropna(axis = 1)
nets.drop(index=nets.index[-1], 
        axis=0, 
        inplace=True)


# In[40]:


nets


# In[41]:


nets = nets.reset_index()


# In[42]:


nets["Change in W/L%"] = 0


# In[43]:


nets


# In[44]:


i = 0
while i < len(nets) - 1:
    if (nets["Coaches"].loc[i].split("("))[0] != (nets["Coaches"].loc[i + 1].split("("))[0]:
        nets["Change in W/L%"].loc[i] = nets["W/L%"].loc[i] - nets["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[45]:


nets


# In[46]:


pd.concat([hawks, celtics, nets])


# In[47]:


hornets = pd.read_csv("Desktop/SAAS/hornets.csv")


# In[48]:


hornets


# In[49]:


hornets.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)


# In[50]:


hornets = hornets[(hornets['Coaches'].str.split('(').str[0] != hornets['Coaches'].shift(1).str.split('(').str[0]) | (hornets['Coaches'].str.split('(').str[0] != hornets['Coaches'].shift(-1).str.split('(').str[0])]


# In[51]:


hornets


# In[52]:


hornets = hornets.dropna(axis = 1)
hornets.drop(index=hornets.index[-1], 
        axis=0, 
        inplace=True)


# In[53]:


hornets


# In[54]:


hornets = hornets.reset_index()


# In[55]:


hornets["Change in W/L%"] = 0


# In[56]:


i = 0
while i < len(hornets) - 1:
    if (hornets["Coaches"].loc[i].split("("))[0] != (hornets["Coaches"].loc[i + 1].split("("))[0]:
        hornets["Change in W/L%"].loc[i] = hornets["W/L%"].loc[i] - hornets["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[57]:


hornets


# In[58]:


bulls = pd.read_csv("Desktop/SAAS/bulls.csv")

bulls.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

bulls = bulls[(bulls['Coaches'].str.split('(').str[0] != bulls['Coaches'].shift(1).str.split('(').str[0]) | (bulls['Coaches'].str.split('(').str[0] != bulls['Coaches'].shift(-1).str.split('(').str[0])]

bulls = bulls.dropna(axis = 1)

bulls = bulls.reset_index()

bulls["Change in W/L%"] = 0

i = 0
while i < len(bulls) - 1:
    if (bulls["Coaches"].loc[i].split("("))[0] != (bulls["Coaches"].loc[i + 1].split("("))[0]:
        bulls["Change in W/L%"].loc[i] = bulls["W/L%"].loc[i] - bulls["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1
        
bulls.drop(index=bulls.index[-1], 
        axis=0, 
        inplace=True)


# In[59]:


bulls


# In[60]:


cavs = pd.read_csv("Desktop/SAAS/cavs.csv")
mavs = pd.read_csv("Desktop/SAAS/mavs.csv")
pistons = pd.read_csv("Desktop/SAAS/pistons.csv")
nuggets = pd.read_csv("Desktop/SAAS/nuggets.csv")
warriors = pd.read_csv("Desktop/SAAS/warriors.csv")


# In[61]:


cavs = pd.read_csv("Desktop/SAAS/cavs.csv")

cavs.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

cavs = cavs[(cavs['Coaches'].str.split('(').str[0] != cavs['Coaches'].shift(1).str.split('(').str[0]) | (cavs['Coaches'].str.split('(').str[0] != cavs['Coaches'].shift(-1).str.split('(').str[0])]

cavs = cavs.dropna(axis = 1)

cavs = cavs.reset_index()

cavs["Change in W/L%"] = 0

i = 0
while i < len(cavs) - 1:
    if (cavs["Coaches"].loc[i].split("("))[0] != (cavs["Coaches"].loc[i + 1].split("("))[0]:
        cavs["Change in W/L%"].loc[i] = cavs["W/L%"].loc[i] - cavs["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[62]:


cavs


# In[63]:


mavs = pd.read_csv("Desktop/SAAS/mavs.csv")

mavs.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

mavs = mavs[(mavs['Coaches'].str.split('(').str[0] != mavs['Coaches'].shift(1).str.split('(').str[0]) | (mavs['Coaches'].str.split('(').str[0] != mavs['Coaches'].shift(-1).str.split('(').str[0])]

mavs = mavs.dropna(axis = 1)

mavs = mavs.reset_index()

mavs["Change in W/L%"] = 0

i = 0
while i < len(mavs) - 1:
    if (mavs["Coaches"].loc[i].split("("))[0] != (mavs["Coaches"].loc[i + 1].split("("))[0]:
        mavs["Change in W/L%"].loc[i] = mavs["W/L%"].loc[i] - mavs["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[64]:


mavs


# In[65]:


pistons = pd.read_csv("Desktop/SAAS/pistons.csv")

pistons.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

pistons = pistons[(pistons['Coaches'].str.split('(').str[0] != pistons['Coaches'].shift(1).str.split('(').str[0]) | (pistons['Coaches'].str.split('(').str[0] != pistons['Coaches'].shift(-1).str.split('(').str[0])]

pistons = pistons.dropna(axis = 1)

pistons = pistons.reset_index()

pistons["Change in W/L%"] = 0

i = 0
while i < len(pistons) - 1:
    if (pistons["Coaches"].loc[i].split("("))[0] != (pistons["Coaches"].loc[i + 1].split("("))[0]:
        pistons["Change in W/L%"].loc[i] = pistons["W/L%"].loc[i] - pistons["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[66]:


pistons


# In[67]:


nuggets = pd.read_csv("Desktop/SAAS/nuggets.csv")

nuggets.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

nuggets = nuggets[(nuggets['Coaches'].str.split('(').str[0] != nuggets['Coaches'].shift(1).str.split('(').str[0]) | (nuggets['Coaches'].str.split('(').str[0] != nuggets['Coaches'].shift(-1).str.split('(').str[0])]

nuggets = nuggets.dropna(axis = 1)

nuggets = nuggets.reset_index()

nuggets["Change in W/L%"] = 0

i = 0
while i < len(nuggets) - 1:
    if (nuggets["Coaches"].loc[i].split("("))[0] != (nuggets["Coaches"].loc[i + 1].split("("))[0]:
        nuggets["Change in W/L%"].loc[i] = nuggets["W/L%"].loc[i] - nuggets["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1
        
nuggets.drop(index=nuggets.index[-1], 
        axis=0, 
        inplace=True)


# In[68]:


nuggets


# In[69]:


warriors = pd.read_csv("Desktop/SAAS/warriors.csv")

warriors.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

warriors = warriors[(warriors['Coaches'].str.split('(').str[0] != warriors['Coaches'].shift(1).str.split('(').str[0]) | (warriors['Coaches'].str.split('(').str[0] != warriors['Coaches'].shift(-1).str.split('(').str[0])]

warriors = warriors.dropna(axis = 1)

warriors = warriors.reset_index()

warriors["Change in W/L%"] = 0

i = 0
while i < len(warriors) - 1:
    if (warriors["Coaches"].loc[i].split("("))[0] != (warriors["Coaches"].loc[i + 1].split("("))[0]:
        warriors["Change in W/L%"].loc[i] = warriors["W/L%"].loc[i] - warriors["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[70]:


warriors


# In[71]:


rockets = pd.read_csv("Desktop/SAAS/rockets.csv")

rockets.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

rockets = rockets[(rockets['Coaches'].str.split('(').str[0] != rockets['Coaches'].shift(1).str.split('(').str[0]) | (rockets['Coaches'].str.split('(').str[0] != rockets['Coaches'].shift(-1).str.split('(').str[0])]

rockets = rockets.dropna(axis = 1)

rockets = rockets.reset_index()

rockets["Change in W/L%"] = 0

i = 0
while i < len(rockets) - 1:
    if (rockets["Coaches"].loc[i].split("("))[0] != (rockets["Coaches"].loc[i + 1].split("("))[0]:
        rockets["Change in W/L%"].loc[i] = rockets["W/L%"].loc[i] - rockets["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[72]:


rockets


# In[73]:


rockets.drop(index=rockets.index[-1], 
        axis=0, 
        inplace=True)


# In[74]:


pacers = pd.read_csv("Desktop/SAAS/pacers.csv")

pacers.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

pacers = pacers[(pacers['Coaches'].str.split('(').str[0] != pacers['Coaches'].shift(1).str.split('(').str[0]) | (pacers['Coaches'].str.split('(').str[0] != pacers['Coaches'].shift(-1).str.split('(').str[0])]

pacers = pacers.dropna(axis = 1)

pacers = pacers.reset_index()

pacers["Change in W/L%"] = 0

i = 0
while i < len(pacers) - 1:
    if (pacers["Coaches"].loc[i].split("("))[0] != (pacers["Coaches"].loc[i + 1].split("("))[0]:
        pacers["Change in W/L%"].loc[i] = pacers["W/L%"].loc[i] - pacers["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[75]:


pacers


# In[76]:


clippers = pd.read_csv("Desktop/SAAS/clippers.csv")

clippers.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

clippers = clippers[(clippers['Coaches'].str.split('(').str[0] != clippers['Coaches'].shift(1).str.split('(').str[0]) | (clippers['Coaches'].str.split('(').str[0] != clippers['Coaches'].shift(-1).str.split('(').str[0])]

clippers = clippers.dropna(axis = 1)

clippers = clippers.reset_index()

clippers["Change in W/L%"] = 0

i = 0
while i < len(clippers) - 1:
    if (clippers["Coaches"].loc[i].split("("))[0] != (clippers["Coaches"].loc[i + 1].split("("))[0]:
        clippers["Change in W/L%"].loc[i] = clippers["W/L%"].loc[i] - clippers["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[77]:


clippers


# In[78]:


lakers = pd.read_csv("Desktop/SAAS/lakers.csv")

lakers.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

lakers = lakers[(lakers['Coaches'].str.split('(').str[0] != lakers['Coaches'].shift(1).str.split('(').str[0]) | (lakers['Coaches'].str.split('(').str[0] != lakers['Coaches'].shift(-1).str.split('(').str[0])]

lakers = lakers.dropna(axis = 1)

lakers = lakers.reset_index()

lakers["Change in W/L%"] = 0

i = 0
while i < len(lakers) - 1:
    if (lakers["Coaches"].loc[i].split("("))[0] != (lakers["Coaches"].loc[i + 1].split("("))[0]:
        lakers["Change in W/L%"].loc[i] = lakers["W/L%"].loc[i] - lakers["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[79]:


lakers


# In[80]:


lakers.drop(index=lakers.index[-1], 
        axis=0, 
        inplace=True)


# In[81]:


grizzlies = pd.read_csv("Desktop/SAAS/grizzlies.csv")

grizzlies.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

grizzlies = grizzlies[(grizzlies['Coaches'].str.split('(').str[0] != grizzlies['Coaches'].shift(1).str.split('(').str[0]) | (grizzlies['Coaches'].str.split('(').str[0] != grizzlies['Coaches'].shift(-1).str.split('(').str[0])]

grizzlies = grizzlies.dropna(axis = 1)

grizzlies = grizzlies.reset_index()

grizzlies["Change in W/L%"] = 0

i = 0
while i < len(grizzlies) - 1:
    if (grizzlies["Coaches"].loc[i].split("("))[0] != (grizzlies["Coaches"].loc[i + 1].split("("))[0]:
        grizzlies["Change in W/L%"].loc[i] = grizzlies["W/L%"].loc[i] - grizzlies["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[82]:


grizzlies


# In[83]:


grizzlies.drop(index=grizzlies.index[-1], 
        axis=0, 
        inplace=True)


# In[84]:


heat = pd.read_csv("Desktop/SAAS/heat.csv")

heat.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

heat = heat[(heat['Coaches'].str.split('(').str[0] != heat['Coaches'].shift(1).str.split('(').str[0]) | (heat['Coaches'].str.split('(').str[0] != heat['Coaches'].shift(-1).str.split('(').str[0])]

heat = heat.dropna(axis = 1)

heat = heat.reset_index()

heat["Change in W/L%"] = 0

i = 0
while i < len(heat) - 1:
    if (heat["Coaches"].loc[i].split("("))[0] != (heat["Coaches"].loc[i + 1].split("("))[0]:
        heat["Change in W/L%"].loc[i] = heat["W/L%"].loc[i] - heat["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[85]:


heat


# In[86]:


heat.drop(index=heat.index[-1], 
        axis=0, 
        inplace=True)


# In[87]:


bucks = pd.read_csv("Desktop/SAAS/bucks.csv")

bucks.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

bucks = bucks[(bucks['Coaches'].str.split('(').str[0] != bucks['Coaches'].shift(1).str.split('(').str[0]) | (bucks['Coaches'].str.split('(').str[0] != bucks['Coaches'].shift(-1).str.split('(').str[0])]

bucks = bucks.dropna(axis = 1)

bucks = bucks.reset_index()

bucks["Change in W/L%"] = 0

i = 0
while i < len(bucks) - 1:
    if (bucks["Coaches"].loc[i].split("("))[0] != (bucks["Coaches"].loc[i + 1].split("("))[0]:
        bucks["Change in W/L%"].loc[i] = bucks["W/L%"].loc[i] - bucks["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[88]:


bucks


# In[89]:


bucks.drop(index=bucks.index[-1], 
        axis=0, 
        inplace=True)


# In[90]:


twolves = pd.read_csv("Desktop/SAAS/twolves.csv")

twolves.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

twolves = twolves[(twolves['Coaches'].str.split('(').str[0] != twolves['Coaches'].shift(1).str.split('(').str[0]) | (twolves['Coaches'].str.split('(').str[0] != twolves['Coaches'].shift(-1).str.split('(').str[0])]

twolves = twolves.dropna(axis = 1)

twolves = twolves.reset_index()

twolves["Change in W/L%"] = 0

i = 0
while i < len(twolves) - 1:
    if (twolves["Coaches"].loc[i].split("("))[0] != (twolves["Coaches"].loc[i + 1].split("("))[0]:
        twolves["Change in W/L%"].loc[i] = twolves["W/L%"].loc[i] - twolves["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[91]:


twolves


# In[92]:


twolves.drop(index=twolves.index[-1], 
        axis=0, 
        inplace=True)


# In[93]:


pelicans = pd.read_csv("Desktop/SAAS/pelicans.csv")

pelicans.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

pelicans = pelicans[(pelicans['Coaches'].str.split('(').str[0] != pelicans['Coaches'].shift(1).str.split('(').str[0]) | (pelicans['Coaches'].str.split('(').str[0] != pelicans['Coaches'].shift(-1).str.split('(').str[0])]

pelicans = pelicans.dropna(axis = 1)

pelicans = pelicans.reset_index()

pelicans["Change in W/L%"] = 0

i = 0
while i < len(pelicans) - 1:
    if (pelicans["Coaches"].loc[i].split("("))[0] != (pelicans["Coaches"].loc[i + 1].split("("))[0]:
        pelicans["Change in W/L%"].loc[i] = pelicans["W/L%"].loc[i] - pelicans["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[94]:


pelicans


# In[95]:


knicks = pd.read_csv("Desktop/SAAS/knicks.csv")

knicks.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

knicks = knicks[(knicks['Coaches'].str.split('(').str[0] != knicks['Coaches'].shift(1).str.split('(').str[0]) | (knicks['Coaches'].str.split('(').str[0] != knicks['Coaches'].shift(-1).str.split('(').str[0])]

knicks = knicks.dropna(axis = 1)

knicks = knicks.reset_index()

knicks["Change in W/L%"] = 0

i = 0
while i < len(knicks) - 1:
    if (knicks["Coaches"].loc[i].split("("))[0] != (knicks["Coaches"].loc[i + 1].split("("))[0]:
        knicks["Change in W/L%"].loc[i] = knicks["W/L%"].loc[i] - knicks["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[96]:


knicks


# In[97]:


knicks.drop(index=knicks.index[-1], 
        axis=0, 
        inplace=True)


# In[98]:


thunder = pd.read_csv("Desktop/SAAS/thunder.csv")

thunder.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

thunder = thunder[(thunder['Coaches'].str.split('(').str[0] != thunder['Coaches'].shift(1).str.split('(').str[0]) | (thunder['Coaches'].str.split('(').str[0] != thunder['Coaches'].shift(-1).str.split('(').str[0])]

thunder = thunder.dropna(axis = 1)

thunder = thunder.reset_index()

thunder["Change in W/L%"] = 0

i = 0
while i < len(thunder) - 1:
    if (thunder["Coaches"].loc[i].split("("))[0] != (thunder["Coaches"].loc[i + 1].split("("))[0]:
        thunder["Change in W/L%"].loc[i] = thunder["W/L%"].loc[i] - thunder["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[99]:


thunder


# In[100]:


thunder.drop(index=thunder.index[-1], 
        axis=0, 
        inplace=True)


# In[101]:


magic = pd.read_csv("Desktop/SAAS/magic.csv")

magic.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

magic = magic[(magic['Coaches'].str.split('(').str[0] != magic['Coaches'].shift(1).str.split('(').str[0]) | (magic['Coaches'].str.split('(').str[0] != magic['Coaches'].shift(-1).str.split('(').str[0])]

magic = magic.dropna(axis = 1)

magic = magic.reset_index()

magic["Change in W/L%"] = 0

i = 0
while i < len(magic) - 1:
    if (magic["Coaches"].loc[i].split("("))[0] != (magic["Coaches"].loc[i + 1].split("("))[0]:
        magic["Change in W/L%"].loc[i] = magic["W/L%"].loc[i] - magic["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[102]:


magic


# In[103]:


magic.drop(index=magic.index[-1], 
        axis=0, 
        inplace=True)


# In[104]:


sixers = pd.read_csv("Desktop/SAAS/sixers.csv")

sixers.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

sixers = sixers[(sixers['Coaches'].str.split('(').str[0] != sixers['Coaches'].shift(1).str.split('(').str[0]) | (sixers['Coaches'].str.split('(').str[0] != sixers['Coaches'].shift(-1).str.split('(').str[0])]

sixers = sixers.dropna(axis = 1)

sixers = sixers.reset_index()

sixers["Change in W/L%"] = 0

i = 0
while i < len(sixers) - 1:
    if (sixers["Coaches"].loc[i].split("("))[0] != (sixers["Coaches"].loc[i + 1].split("("))[0]:
        sixers["Change in W/L%"].loc[i] = sixers["W/L%"].loc[i] - sixers["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[105]:


sixers


# In[106]:


sixers.drop(index=sixers.index[-1], 
        axis=0, 
        inplace=True)


# In[107]:


suns = pd.read_csv("Desktop/SAAS/suns.csv")

suns.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

suns = suns[(suns['Coaches'].str.split('(').str[0] != suns['Coaches'].shift(1).str.split('(').str[0]) | (suns['Coaches'].str.split('(').str[0] != suns['Coaches'].shift(-1).str.split('(').str[0])]

suns = suns.dropna(axis = 1)

suns = suns.reset_index()

suns["Change in W/L%"] = 0

i = 0
while i < len(suns) - 1:
    if (suns["Coaches"].loc[i].split("("))[0] != (suns["Coaches"].loc[i + 1].split("("))[0]:
        suns["Change in W/L%"].loc[i] = suns["W/L%"].loc[i] - suns["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[108]:


suns


# In[109]:


suns.drop(index=suns.index[-1], 
        axis=0, 
        inplace=True)


# In[110]:


trailblazers = pd.read_csv("Desktop/SAAS/trailblazers.csv")

trailblazers.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

trailblazers = trailblazers[(trailblazers['Coaches'].str.split('(').str[0] != trailblazers['Coaches'].shift(1).str.split('(').str[0]) | (trailblazers['Coaches'].str.split('(').str[0] != trailblazers['Coaches'].shift(-1).str.split('(').str[0])]

trailblazers = trailblazers.dropna(axis = 1)

trailblazers = trailblazers.reset_index()

trailblazers["Change in W/L%"] = 0

i = 0
while i < len(trailblazers) - 1:
    if (trailblazers["Coaches"].loc[i].split("("))[0] != (trailblazers["Coaches"].loc[i + 1].split("("))[0]:
        trailblazers["Change in W/L%"].loc[i] = trailblazers["W/L%"].loc[i] - trailblazers["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[111]:


trailblazers


# In[112]:


trailblazers.drop(index=trailblazers.index[-1], 
        axis=0, 
        inplace=True)


# In[113]:


kings = pd.read_csv("Desktop/SAAS/kings.csv")

kings.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

kings = kings[(kings['Coaches'].str.split('(').str[0] != kings['Coaches'].shift(1).str.split('(').str[0]) | (kings['Coaches'].str.split('(').str[0] != kings['Coaches'].shift(-1).str.split('(').str[0])]

kings = kings.dropna(axis = 1)

kings = kings.reset_index()

kings["Change in W/L%"] = 0

i = 0
while i < len(kings) - 1:
    if (kings["Coaches"].loc[i].split("("))[0] != (kings["Coaches"].loc[i + 1].split("("))[0]:
        kings["Change in W/L%"].loc[i] = kings["W/L%"].loc[i] - kings["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[114]:


kings


# In[115]:


kings.drop(index=kings.index[-1], 
        axis=0, 
        inplace=True)


# In[116]:


spurs = pd.read_csv("Desktop/SAAS/spurs.csv")

spurs.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

spurs = spurs[(spurs['Coaches'].str.split('(').str[0] != spurs['Coaches'].shift(1).str.split('(').str[0]) | (spurs['Coaches'].str.split('(').str[0] != spurs['Coaches'].shift(-1).str.split('(').str[0])]

spurs = spurs.dropna(axis = 1)

spurs = spurs.reset_index()

spurs["Change in W/L%"] = 0

i = 0
while i < len(spurs) - 1:
    if (spurs["Coaches"].loc[i].split("("))[0] != (spurs["Coaches"].loc[i + 1].split("("))[0]:
        spurs["Change in W/L%"].loc[i] = spurs["W/L%"].loc[i] - spurs["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[117]:


spurs


# In[118]:


all_coaching_changes = pd.concat([hawks, celtics, nets, hornets, bulls, cavs, mavs, pistons, nuggets, warriors, rockets, pacers, clippers, lakers, grizzlies, heat, bucks, twolves, pelicans, knicks, thunder, magic, sixers, suns, trailblazers, kings, spurs])
all_coaching_changes = all_coaching_changes.reset_index()
all_coaching_changes


# In[125]:


raptors = pd.read_csv("Desktop/SAAS/raptors.csv")

raptors.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

raptors = raptors[(raptors['Coaches'].str.split('(').str[0] != raptors['Coaches'].shift(1).str.split('(').str[0]) | (raptors['Coaches'].str.split('(').str[0] != raptors['Coaches'].shift(-1).str.split('(').str[0])]

raptors = raptors.dropna(axis = 1)

raptors = raptors.reset_index()

raptors["Change in W/L%"] = 0

i = 0
while i < len(raptors) - 1:
    if (raptors["Coaches"].loc[i].split("("))[0] != (raptors["Coaches"].loc[i + 1].split("("))[0]:
        raptors["Change in W/L%"].loc[i] = raptors["W/L%"].loc[i] - raptors["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[126]:


raptors


# In[127]:


raptors.drop(index=raptors.index[0], 
        axis=0, 
        inplace=True)


# In[128]:


raptors


# In[129]:


raptors = raptors.reset_index()


# In[130]:


raptors


# In[131]:


jazz = pd.read_csv("Desktop/SAAS/jazz.csv")

jazz.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

jazz = jazz[(jazz['Coaches'].str.split('(').str[0] != jazz['Coaches'].shift(1).str.split('(').str[0]) | (jazz['Coaches'].str.split('(').str[0] != jazz['Coaches'].shift(-1).str.split('(').str[0])]

jazz = jazz.dropna(axis = 1)

jazz = jazz.reset_index()

jazz["Change in W/L%"] = 0

i = 0
while i < len(jazz) - 1:
    if (jazz["Coaches"].loc[i].split("("))[0] != (jazz["Coaches"].loc[i + 1].split("("))[0]:
        jazz["Change in W/L%"].loc[i] = jazz["W/L%"].loc[i] - jazz["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[132]:


jazz


# In[133]:


jazz.drop(index=jazz.index[0], 
        axis=0, 
        inplace=True)
jazz = jazz.reset_index()


# In[134]:


jazz


# In[135]:


wizards = pd.read_csv("Desktop/SAAS/wizards.csv")

wizards.drop(['Lg', 'Finish', 'SRS', 'Pace', 'Rel Pace', 'ORtg', 'Rel ORtg', 'DRtg', 'Rel DRtg', 'Playoffs', 'Top WS'], axis = 1, inplace = True)

wizards = wizards[(wizards['Coaches'].str.split('(').str[0] != wizards['Coaches'].shift(1).str.split('(').str[0]) | (wizards['Coaches'].str.split('(').str[0] != wizards['Coaches'].shift(-1).str.split('(').str[0])]

wizards = wizards.dropna(axis = 1)

wizards = wizards.reset_index()

wizards["Change in W/L%"] = 0

i = 0
while i < len(wizards) - 1:
    if (wizards["Coaches"].loc[i].split("("))[0] != (wizards["Coaches"].loc[i + 1].split("("))[0]:
        wizards["Change in W/L%"].loc[i] = wizards["W/L%"].loc[i] - wizards["W/L%"].loc[i + 1]
        i = i + 1
    else:
        i = i + 1


# In[136]:


wizards


# In[137]:


wizards.drop(index=wizards.index[0], 
        axis=0, 
        inplace=True)
wizards = wizards.reset_index()
wizards


# In[139]:


all_coaching_changes = pd.concat([hawks, celtics, nets, hornets, bulls, cavs, mavs, pistons, nuggets, warriors, rockets, pacers, clippers, lakers, grizzlies, heat, bucks, twolves, pelicans, knicks, thunder, magic, sixers, suns, trailblazers, kings, spurs, raptors, jazz, wizards])


# In[140]:


all_coaching_changes

