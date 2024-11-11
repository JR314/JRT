#This program still need to update and fix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets # In the real test, we replace the two csv with the others
anime_df = pd.read_csv("anime.csv")
user_ratings_df = pd.read_csv("user_ratings.csv")

# Display the first few rows of each DataFrame
print("Anime Data:")
print(anime_df.head())
print("\nUser Ratings Data:")
print(user_ratings_df.head())

# Check for missing values
print("Missing values in Anime Data:")
print(anime_df.isnull().sum())

print("\nMissing values in User Ratings Data:")
print(user_ratings_df.isnull().sum())

# Basic statistics for numeric columns
print("\nAnime Data Summary:")
print(anime_df.describe())

print("\nUser Ratings Data Summary:")
print(user_ratings_df.describe())

#distribute the anime scores
plt.figure(figsize=(10, 5))
sns.histplot(anime_df['average_rating'], bins=20, kde=True)
plt.title("Distribution of Anime Scores")
plt.xlabel("Average Rating")
plt.ylabel("Frequency")
plt.show()

#distribute the release years
plt.figure(figsize=(10, 5))
sns.histplot(anime_df['release_year'], bins=20, kde=True)
plt.title("Distribution of Anime Release Years")
plt.xlabel("Release Year")
plt.ylabel("Frequency")
plt.show()

#distribute number of episodes
plt.figure(figsize=(10, 5))
sns.histplot(anime_df['episodes'], bins=20, kde=True)
plt.title("Distribution of Episode Counts")
plt.xlabel("Number of Episodes")
plt.ylabel("Frequency")
plt.show()

# Split genres and explode the DataFrame
anime_df['genre'] = anime_df['genre'].fillna('')
genres = anime_df['genre'].str.split(',').explode()
genres = genres[genres != '']  # Remove empty strings if any

# Plot genre popularity
plt.figure(figsize=(12, 6))
sns.countplot(y=genres, order=genres.value_counts().index)
plt.title("Genre Popularity")
plt.xlabel("Count")
plt.ylabel("Genre")
plt.show()

# Plot Average rating
plt.figure(figsize=(10, 5))
sns.histplot(user_ratings_df.groupby('user_id')['rating'].mean(), bins=20, kde=True)
plt.title("Distribution of Average Ratings per User")
plt.xlabel("Average Rating")
plt.ylabel("Frequency")
plt.show()

# Plot Completion Status
plt.figure(figsize=(10, 5))
sns.countplot(data=user_ratings_df, x='completion_status')
plt.title("Completion Status Distribution")
plt.xlabel("Completion Status")
plt.ylabel("Count")
plt.show()
