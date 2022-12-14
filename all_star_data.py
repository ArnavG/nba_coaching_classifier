#!/usr/bin/env python
# coding: utf-8

# In[1]:



# coding: utf-8

 
# Credits goes to https://github.com/SpecCRA/nba_data_scrapers repo

from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import time
import sys


# # Web page scraping

 


# Create url templates for each kind of stats
per_g_url_template = "https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
adv_url_template = "https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
tot_url_template = "https://www.basketball-reference.com/leagues/NBA_{year}_totals.html"
per_36m_url_template = "https://www.basketball-reference.com/leagues/NBA_{year}_per_minute.html"
per_100p_url_template = "https://www.basketball-reference.com/leagues/NBA_{year}_per_poss.html"

# Put all the URL templates into a list
url_template_list = [per_g_url_template, adv_url_template, tot_url_template,
                     per_36m_url_template, per_100p_url_template]


# Ask user to input start and end years
# Also checks to see if entry is a number
try:
    user_start_year = int(input("Enter start year in YYYY format: "))
except:
    print('Enter a valid 4 digit year.')

try:
    user_end_year = int(input("Enter end year in YYYY format: "))
except:
    print('Enter a valid 4 digit year.')


# Check if end year is after start year
if user_end_year >= user_start_year:
    print('Year range accepted.')
else:
    print('Year range is unacceptable.')

# Check if formats are in proper YYYY format
def check_year(user_input_year):
    if user_input_year > 999 and user_input_year < 10000: # Then check if it's 4 digits
        print('Year format accepted.')
    else:
        print('Enter a valid 4 digit year.')
        sys.exit()

# Check both entered years for formatting
check_year(user_start_year)
check_year(user_end_year)

# Create empty lists to store data before appending to Dataframe
column_headers = []
player_data = []
# Create empty DataFrame for following functions to fill
df = pd.DataFrame()


# Empty DataFrames for each set of pages
df_adv = pd.DataFrame()
df_per_g = pd.DataFrame()
df_tot = pd.DataFrame()
df_per_36m = pd.DataFrame()
df_per_100p = pd.DataFrame()

# Create df_list of DataFrames for looping
df_list = [df_per_g, df_adv, df_tot, df_per_36m, df_per_100p]



# Get column headers from each page
# Assigns a new list of column headers each time this is called
def get_column_headers(soup):
    headers = []
    for th in soup.find('tr').findAll('th'):
        #print th.getText()
        headers.append(th.getText())
    #print headers # this line was for a bug check
    # Assign global variable to headers gathered by function
    return headers
#column_headers = [th.getText() for th in soup.find('tr').findAll('th')]


# Function to get player data from each page
def get_player_data(soup):
    # Temporary list within function to store data
    temp_player_data = []

    data_rows = soup.findAll('tr')[1:] # skip first row
    for i in range(len(data_rows)): # loop through each table row
        player_row = [] # empty list for each player row
        for td in data_rows[i].findAll('td'):
            player_row.append(td.getText()) # append separate data points
        temp_player_data.append(player_row) # append player row data
    return temp_player_data


# This function scrapes the stats data of one page and returns it in a DataFrame
def scrape_page(url):
    r = requests.get(url) # get the url
    soup = BeautifulSoup(r.text, 'html.parser') # Create BS object

    # call function to get column headers
    column_headers = get_column_headers(soup)

    # call function to get player data
    player_data = get_player_data(soup)

    #data to DataFrame
    # Skip first value of column headers, 'Rk'
    df = pd.DataFrame(player_data, columns = column_headers[1:])

    return df


def get_season(input_year):
    first_yr = input_year - 1
    season = str(first_yr) + "-" + str(input_year)[2:]
    return season


# This function drops empty rows an columns, drops duplicates, and changes
# % character in columns
def gen_cleaning(df):
    # Convert values to numeric values first
    df = df.apply(pd.to_numeric, errors = 'ignore')

    # Drop columns with no data
    df.dropna(axis = 1, how = "all", inplace = True)

    # Drop rows with no data
    df.dropna(axis = 0, how = "all", inplace = True)

    # Remove duplicates player inputs; ie. players who were traded
    # I only kept the TOT per game season values
    #df.drop_duplicates(["Player"], keep = "first", inplace = True)

    # Change % symbol to _perc
    df.columns = df.columns.str.replace('%', '_perc')

    return df


# This function scrapes player data from multiple pages by start and end years
def scrape_pages(url_template, start_year, end_year, output_df):
    count = 0
    for year in range(start_year, (end_year+1)):
        url = url_template.format(year = year) # grab URL per year
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html5lib') # Create soup item

        # Check to grab column headers
        if count == 0: # only append column headers once
            columns = get_column_headers(soup)
            count += 1

        # grab player data for each year
        player_data = get_player_data(soup)

        # Create temporary DataFrame first for each year
        # Duplicates are removed before putting into bigger DataFrame
        # These duplicates come from players playing on multiple teams in one season
        # This script only keeps the TOT output as Tm
        year_df = pd.DataFrame(player_data, columns = columns[1:])
        year_df.drop_duplicates(['Player'], keep = 'first', inplace = True)
        year_df.insert(0, 'Season', get_season(year))  #season year column

        # Append to big DataFrame for detailed cleaning
        output_df = output_df.append(year_df, ignore_index = True)

    # Do common, general cleaning practices
    output_df = gen_cleaning(output_df)

    return output_df



# Fill each DataFrame with data scraped from their respective pages
# Each print statement is a check for if any pages or functions give issues
# Added timer to check how long this was taking

start = time.time()

df_per_g = scrape_pages(per_g_url_template, user_start_year, user_end_year, df_per_g)
print("Finished per g")
df_adv = scrape_pages(adv_url_template, user_start_year, user_end_year, df_adv)
print("Finished adv")
df_tot = scrape_pages(tot_url_template, user_start_year, user_end_year, df_tot)
print("Finished tots")
df_per_36m = scrape_pages(per_36m_url_template, user_start_year, user_end_year, df_per_36m)
print("Finished per 36m")
df_per_100p = scrape_pages(per_100p_url_template, user_start_year, user_end_year, df_per_100p)
print("Finished per 100 possessions")

end = time.time()
print("Time elapsed :" +str((end - start) / 60) + " minutes")


# # Data Auditing and Cleaning

# Check all column names to see what needs to be cleaned

print("totals")
print(list(df_tot))
print("per game")
print(list(df_per_g))
print("per 36 minutes")
print(list(df_per_36m))
print("advanced")
print(list(df_adv))
print("per 100p")
print(list(df_per_100p))



# Label columns properly by adding "_tot" to the end of some column values
df_tot.columns.values[[7, 8 , 9, 11, 12, 14, 15, 18, 19]] = [df_tot.columns.values[[7, 8 , 9, 11, 12, 14, 15, 18, 19]][col] + "_tot" for col in range(9)]

df_tot.columns.values[21:30] = [df_tot.columns.values[21:30][col] + "_tot" for col in range(9)]



# Check column titles again
list(df_tot)



# drop _perc columns from per_g and per_36m
# Never mind, drop duplicates later on
# Add _per_g and _per_36m to column values
# Add _per_G to some values in df_per_g
df_per_g.columns.values[[7, 8 , 9, 11, 12, 14, 15, 18, 19]] = [df_per_g.columns.values[[7, 8 , 9, 11, 12, 14, 15, 18, 19]][col] + "_per_G" for col in range(9)]

df_per_g.columns.values[21:29] = [df_per_g.columns.values[21:30][col] + "_per_G" for col in range(8)]

# Rename PS/G to PTS_per_G
df_per_g.rename(columns={'PS/G': 'PTS_per_G'}, inplace = True)



df_per_36m.columns.values[[7, 8, 9, 11, 12, 14, 15, 18, 19]]


 


# Check if proper values were changed
list(df_per_g)


 


# Add per_36m to its column values
df_per_36m.columns.values[[8, 9, 11, 12, 14, 15, 17, 18]] = [df_per_36m.columns.values[[8, 9, 11, 12, 14, 15, 17, 18]][col] + "_per_36m" for col in range(8)]

df_per_36m.columns.values[20:30] = [df_per_36m.columns.values[20:30][col] + "_per_36m"                                    for col in range(9)]


 


# Check columns were changed properly
list(df_per_36m)


 


# Add per_100p to per 100 possession column values
list_of_changes = ['FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL',                    'BLK', 'TOV', 'PF', 'PTS']
# Grab a list of current column names
column_values = list(df_per_100p.columns.values)

# Create a list for updated column names
updated_columns = []

# Loop through original column values to see what to replace
for value in column_values:
    if value in list_of_changes:
        updated_columns.append(value + '_per_100p')
    else:
        updated_columns.append(value)

# Update column values
df_per_100p.columns = updated_columns


 


# Check if columns are properly named
list(df_per_100p)


 


# Find where '\xa0' columns are for removal
print(df_adv.columns[-5])
print(df_adv.columns[19])


 


# Drop '\xa0' columns, last one first
#df_adv.drop(df_adv.columns[-5], axis = 1, inplace = True)
#df_adv.drop(df_adv.columns[19], axis = 1, inplace = True)


 


list(df_adv)


 


df_adv.rename(columns = {'WS/48' : 'WS_per_48'}, inplace = True)


 


# Check to see if columns were dropped properly
list(df_adv)


 


# Merge dataframes later on season, player name, and team
# Order of merges: tots, per_g, per_36m, adv
# DFs: df_tot, df_per_g, df_per_36m, df_adv
# Common things: Season, Player, Pos, Age, Tm, G


 


df_all = pd.merge(df_tot, df_per_g, how = "left",
                  on = ['Season', 'Player', 'Pos', 'Age', 'Tm', 'G', 'GS', 'FT_perc',
                        '3P_perc', '2P_perc', 'FG_perc', 'eFG_perc'])


 


df_all = pd.merge(df_all, df_per_36m, how = "left",
                  on = ['Season', 'Player', 'Pos', 'Age', 'Tm', 'G', 'GS', 'FT_perc',
                        '3P_perc', '2P_perc', 'FG_perc'])


 


df_all = pd.merge(df_all, df_adv, how = "left",
                  on = ['Season', 'Player', 'Pos', 'Age', 'Tm', 'G'])


 


df_all = pd.merge(df_all, df_per_100p, how = "left",
                  on = ['Season', 'Player', 'Pos', 'Age', 'Tm', 'G', 'GS',
                        'FG_perc', '3P_perc', '2P_perc', 'FT_perc'])


 


# Check columns to make sure they're all right
list(df_all)


 


# Try to drop duplicate MP columns
list(df_all.drop(['MP_x', 'MP_y'], axis = 1))


 


df_all.drop(['MP_x', 'MP_y'], axis = 1, inplace = True)


 


# Final check of columns
list(df_all)


 


# First check length of dataframe
print(len(df_all))


 


# Fill Null values with 0
df_all.fillna(0, inplace = True)


 


# Address ambiguous positions and combination positions
df = df_all.groupby(['Pos'])['Pos'].nunique()
df


 


# Remove where 'Pos' value is 0
df_all = df_all[df_all['Pos'] != 0]

# Then check df_all length again
print(len(df_all))


 


# I think the PG-SF and C-SF positions are mistakes
# Check the value to see the player
df_all[df_all['Pos'] == 'C-SF']


 


# Check Bobby Jones' actual, commonly played position
df_all[df_all['Player'] == 'Bobby Jones']


 


# Create list of dual positions in DataFrame
# Create empty DataFrame to audit dual position values
column_names = list(df_all.columns.values)
dual_pos_rows = []
df_dual_pos = pd.DataFrame(columns = column_names)


 


# Gather all the dual positions by seeing which ones have a dash
for pos in df_all['Pos']:
    if "-" in pos:
        if pos not in dual_pos_rows:
            dual_pos_rows.append(pos)


 


# Append all dual position rows to a new DataFrame for auditing
for pos in dual_pos_rows:
    df_dual_pos = df_dual_pos.append(df_all[df_all['Pos'] == pos],
                                     ignore_index = True)


 


df_dual_pos
# It looks like all these players moved teams before
# Certain players have multiple positions or changed positions


 


df_dual_pos.groupby(['Player']).size().reset_index(name = 'Count').sort_values(['Count'], ascending = False).head(n=10)


 


# Check what is going on with some players with multiple positions
# 2018-19 Seasons_all[df_all['Player'] == 'Allen Iverson*'].groupby(['Pos']).size().iloc[0]


 


# Use dictionary as key to replace 'Pos' values in the big DataFrame
most_common_pos = {}
# Saves a dictionary of player names with equally common positions
two_common_pos = {}
# PG, SG, SF, PF, C are 1-5, respectively
pos_key = {'PG': '1', 'SG': '2', 'SF': '3', 'PF': '4', 'C': '5','F':'2'}

# Side note: This makes Tim Duncan a center

def grab_most_common_pos(df, pos_dict):
    # Loop through a dataframe and assign names and most common positions to a dictionary
    for index, row in df[['Player', 'Pos']].iterrows():
        player_name = row.iloc[0] # Assign player name to variable
        # subset position dataframe to a player
        pos_df = df[df['Player'] == player_name].groupby('Pos').size()        .reset_index(name = 'Count')        .sort_values(['Count'], ascending = False)

        #dual_pos_dict = {} # Store dict of dual positions

        pos = pos_df.iloc[0][0] # Assign first position to variable
        second_pos = ''

        # Fill in second position if it exists
        if len(pos_df) > 1:
            second_pos = pos_df.iloc[1][0]

        # Check is player has a second common position
        # I don't know what to do in this situation yet
        #if pos_df.iloc[0][1] == pos_df.iloc[1][1]:
        #    dual_pos_dict['First position'] = pos
        #    dual_pos_dict['Second position'] = second_pos
        #    two_common_pos[player_name] = dual_pos_dict
        #print(player_name)

        if player_name not in pos_dict.keys(): # Check if name exists first
            pos_dict[player_name] = pos

#return pos_dict

def clean_pos(df, pos_dict):
    # Loop through rows to check players' positions
    grab_most_common_pos(df, pos_dict)

    # If the most common position is a dual position, take the first one
    for name, pos in pos_dict.items():
        if '-' in pos:
            index = pos.find('-')
            pos_dict[name] = pos[:index]
        else:
            continue

    # Change pos_dict values to 1-5 from key
    for key, value in pos_dict.items():
        pos_dict[key] = pos_key[value]

    # Return DataFrame with cleaned positions
    return df


 


# This function takes in a DataFrame and adds a new column with Rounded Position values
def assign_pos(df, pos_dict):
    # Add a Rounded_Pos column and fill it from pos_dict
    df['Rounded_Pos'] = ''

    for name, pos in pos_dict.items():
        df.Rounded_Pos[df['Player'] == name] = pos


 


clean_pos(df_all, most_common_pos)


 


# assign_pos(df_all, most_common_pos)


# # Write to csv file for use

 


# Create a DataFrame with top 25 single season scorers
#df_top_25_scorers = df_all.sort_values('PTS_per_G', ascending = False).head(n=25)

# Create a DataFrame with top 50 single season scorers
#df_top_50_scorers = df_all.sort_values('PTS_per_G', ascending = False).head(n=50)


 


# Write to CSV files and DONE!
file_name = 'player_data_' + str(user_start_year) + '-' + str(user_end_year) + '.csv'
#print(file_name)
df_all.to_csv(file_name, encoding = 'utf-8', index = False)


 


#df_top_50_scorers.to_csv("bref_1981_2017_top_50_season_scorers.csv", encoding = "utf-8", index = False)


# In[2]:


df_all


# In[ ]:




