"""
dataset.py - Centralized Data Pipeline (Baseline Configuration)
Restores the original chronological splitting mechanics and title-only history tokens.
"""

import os
import pandas as pd

def clean_and_map_genres(genre_str):
    """Maps raw comma-separated integer codes or string configurations to a clean text list."""
    genre_map = {
        0: "Action", 1: "Adventure", 2: "Animation", 3: "Children", 4: "Comedy",
        5: "Crime", 6: "Documentary", 7: "Drama", 8: "Fantasy", 9: "Film-Noir",
        10: "Horror", 11: "Musical", 12: "Mystery", 13: "Romance", 14: "Sci-Fi",
        15: "Thriller", 16: "War", 17: "Western"
    }
    
    if pd.isna(genre_str) or str(genre_str).strip() == "":
        return "Unknown"
        
    mapped_genres = []
    for item in str(genre_str).split(","):
        cleaned_item = item.strip()
        if cleaned_item.isdigit():
            mapped_genres.append(genre_map.get(int(cleaned_item), "Unknown"))
        else:
            mapped_genres.append(cleaned_item)
            
    return ", ".join(mapped_genres)

def prepare_and_split_data():
    """Builds clean text contexts and partitions data chronologically on an 80/20 split."""
    print("Loading raw transactional records from data directory...")
    
    movies = pd.read_csv(os.path.join("data", "movies.csv"))
    users = pd.read_csv(os.path.join("data", "users.csv"))
    ratings = pd.read_csv(os.path.join("data", "ratings.csv"))

    movies['movie_genres'] = movies['movie_genres'].apply(clean_and_map_genres)
    users['gender_str'] = users['user_gender'].map({True: 'Male', False: 'Female'})

    # Item Tower: Holds rich text metadata
    movies['movie_context'] = "Movie: " + movies['movie_title'] + " | Genres: " + movies['movie_genres']

    # Filter explicit high-engagement positive preferences
    positive_ratings = ratings[ratings['user_rating'] >= 4.0].copy()
    positive_ratings = positive_ratings.sort_values(['user_id', 'timestamp'])
    
    title_map = movies.set_index('movie_id')['movie_title'].to_dict()
    raw_titles = positive_ratings['movie_id'].map(title_map).astype(str)
    
    # Baseline history token strategy (titles only)
    positive_ratings['movie_history_token'] = raw_titles.apply(lambda x: x.split(" (")[0].strip())

    # Build sequence history vectors
    history_1 = positive_ratings.groupby('user_id')['movie_history_token'].shift(1).fillna("None")
    history_2 = positive_ratings.groupby('user_id')['movie_history_token'].shift(2).fillna("None")
    history_3 = positive_ratings.groupby('user_id')['movie_history_token'].shift(3).fillna("None")
    positive_ratings['history_context'] = history_1 + " | " + history_2 + " | " + history_3

    # Generate User Profile strings
    full_dataset = positive_ratings.merge(users, on='user_id').merge(movies, on='movie_id')
    full_dataset['user_context'] = (
        "User Profile: " + full_dataset['gender_str'] + \
        ", Age Group: " + full_dataset['bucketized_user_age'].astype(int).astype(str) + \
        ", History: " + full_dataset['history_context']
    )
    
    # Chronological splitting
    full_dataset = full_dataset.sort_values('timestamp')
    split_idx = int(len(full_dataset) * 0.8)
    
    train_df = full_dataset.iloc[:split_idx]
    test_df = full_dataset.iloc[split_idx:]
    
    output_movies_path = os.path.join("data", "processed_movies.csv")
    movies[['movie_title', 'movie_genres', 'poster_url', 'movie_context']].to_csv(output_movies_path, index=False)
    
    return train_df, test_df, movies