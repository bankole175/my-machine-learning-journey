from os import name

import numpy as np
import pandas as pd

# series
# data = [100, 102, 104, 200, 202]
# series = pd.Series(data, index=["a", "b", "c", "d", "e"])

# to locate a data with label
# print(series.loc["a"])

# to locate a data with index
# print(series.iloc[0])

# to get values greater than 200 in the seires
# print(series[series > 200])

# ________________________________________________
# dictionary

# calories = {'Day 1': 1750, 'Day 2': 2100, 'Day 3': 1700}
# series = pd.Series(calories)
# print(series[series > 2000])

#________________________________________________
# dataframe

# data = {
#     "Name": ["Spongebob", "Patrick", "Squidward"],
#     "age": [30, 37, 23]
# }
# df = pd.DataFrame(data, index=['Employee 1', 'Employee 2', 'Employee 3'])
#
# # Add a new column (lowercase 'job')
# df["job"] = ["Cook", "N/A", "Sing"]
#
# # Create new row with matching column names (lowercase)
# # and a matching index style
# new_row = pd.DataFrame(
#     [{"Name": "Sandy", "age": 28, "job": "Engineer"}],
#     index=["Employee 4"]
# )
#
# df = pd.concat([df, new_row])

# print(df)

df = pd.read_csv('data.csv')

# selection by row
# print(df.loc["Charizard": "Blastoise"], ["Height", "Weight"])
# print(df.iloc[0:11:2, 0:3])

# pokemon = input("Enter Pokemon Name: ")
# try:
#     print(df.loc[pokemon])
# except KeyError:
#     print("Pokemon not found in DataFrame")


# __________________________________
# filtering
# tall_pokermon = df[df["Height"] >= 2]
# heavy_pokermon = df[df["Weight"] > 100]
# lengadary_pokermon = df[df["Legendary"] == 1]
# water_pokermon = df[(df["Type1"] == "Water") | (df["Type2"] == "Water")]
# ff_pokeon = df[(df["Type1"] == "Fire") & (df["Type2"] == "Flying")]

# __________________________________________
# Aggregate functions for whole dataframe
# print(df.mean(numeric_only=True))
# print(df.sum(numeric_only=True))
# print(df.min(numeric_only=True))
# print(df.max(numeric_only=True))
# print(df.count())

# ______________________________________________
# Aggregate functions for single column
# print(df["Height"].mean())
# print(df["Height"].sum())
# print(df["Height"].min())
# print(df["Height"].max())
# print(df["Height"].count())

#________________________________________________
# Grouping dataframe
# group = df.groupby("Type1")
# print(group["Height"].mean())
# print(group["Height"].sum())
# print(group["Height"].max())
# print(group["Height"].min())
# print(group["Height"].count())

# ________________________________________________
# Data cleaning
# drop irrelevant columns
# df = df.drop(columns=["Legendary", "No"])
# print(df)

# handle missing data
# df = df.dropna(subset=['Type2'])
# print(df.to_string())

# replace missing value
# df = df.fillna({'Type2': 'None'})
# print(df.to_string())

#fix inconsistent value
# df['Type1'] = df['Type1'].replace({'Grass': 'GRASS', 'Fire': 'FIRE', 'Water': 'WATER'})
# print(df.to_string())

# standardize text
# df["Name"] = df["Name"].str.lower()
# print(df.to_string())


# fix data types
# df["Legendary"] = df['Legendary'].astype(bool)
# print(df.to_string())

#how to remove duplicate value
# df = df.drop_duplicates()
# print(df.to_string())

# https://www.kaggle.com/code/bhankypsalm/exercise-summary-functions-and-maps/edit

# To find the most frequent reviewers, you use the value_counts() method on the taster_twitter_handle column.
# reviews_written = reviews.groupby('taster_twitter_handle').size()
# Why your previous answer was marked "Incorrect":Order: value_counts() sorts by the counts (descending).groupby().size(): Sorts by the index labels (alphabetical Twitter handles) by default.Metadata: value_counts() creates a Series with a specific name property that can sometimes clash with automated checkers.

# To find the highest-rated wine for every price point, you should use groupby on the price column and then select the maximum of the points column.
# Breakdown of the code:groupby('price'): This groups all reviews that share the same price into buckets.['points'].max(): For each of those price buckets, we look only at the points column and pick the highest (maximum) value..sort_index(): Since the index of this new Series is the price, sorting the index ensures the result starts at the lowest price ($4.0) and ends at the highest ($3300.0).Why use sort_index() instead of sort_values()?sort_values() would sort the result by the points (e.g., 80 points to 100 points).sort_index() sorts by the labels (the prices), which is exactly what the prompt asked for.

# To find the minimum and maximum prices for each wine variety, you use the .groupby() method combined with .agg() (aggregate).
# price_extremes = reviews.groupby('variety').price.agg(['min', 'max'])
# Why this works:.groupby('variety'): Groups the data so that every unique wine variety (like Pinot Noir or Chardonnay) has its own bucket..price: Tells Pandas we only care about the price column for our calculation..agg(['min', 'max']): This is a powerful shortcut that runs both the min() and max() functions at the same time.Result: It automatically returns a DataFrame where the index is the variety and the columns are labeled "min" and "max".

# To sort your previous result by multiple criteria, you pass a list of column names to .sort_values().
# sorted_varieties = price_extremes.sort_values(by=['min', 'max'], ascending=False)


# To calculate the average score per reviewer, use groupby on the taster_name column and calculate the mean of the points.points
# Why this works:groupby('taster_name'): Groups the data by the reviewer's full name..points.mean(): Isolates the points column and calculates the average for each person.Result: You get a Series where the index is the reviewer and the value is their average score.

# To find the most common combinations, you group by both columns and count the occurrences. Because you need the results sorted by count, value_counts() is the most direct way to do this.
# country_variety_counts = reviews.groupby(['country', 'variety']).size().sort_values(ascending=False)
# Why this works:['country', 'variety']: Passing a list to groupby creates a MultiIndex. Each entry in the Series is now identified by a pair (e.g., (US, Pinot Noir))..size(): This counts the number of rows for every unique combination found in the dataset..sort_values(ascending=False): This ensures the most frequent pairings (like US Pinot Noir or Italian Red Blends) appear at the top.