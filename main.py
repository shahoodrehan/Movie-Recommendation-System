import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommendationSystem:
    def __init__(self, imdb_dataset_path, user_history_path):
        self.imdb_dataset = self.load_imdb_dataset(imdb_dataset_path)
        self.user_history = self.load_user_history(user_history_path)
        self.logged_in_user = None
        self.tf_idf_vectorizer = TfidfVectorizer()

    def load_imdb_dataset(self, path):
        return pd.read_csv(path)

    def load_user_history(self, path):
        return pd.read_csv(path, encoding='latin1')

    def login(self, username, password, credentials_file):
        with open(credentials_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                stored_username, stored_password = row
                if stored_username == username and stored_password == password:
                    print("Login successful.")
                    self.logged_in_user = username
                    return True
        print("Invalid username or password.")
        return False

    def recommend_movies(self):
        if self.logged_in_user:
            user_watched_movies = self.user_history[self.user_history['Username'] == self.logged_in_user]
            user_features = user_watched_movies[['director_name', 'genres', 'plot_keywords', 'duration',
                                                  'actor_1_name', 'actor_2_name', 'actor_3_name']].fillna('')
            
            # Convert numeric columns to string
            user_features['duration'] = user_features['duration'].astype(str)

            # Concatenate all features into a single string for each movie
            user_features_str = user_features.apply(lambda x: ' '.join(x.astype(str)), axis=1)

            # Fit TF-IDF vectorizer on IMDb dataset
            self.tf_idf_vectorizer.fit(self.imdb_dataset.apply(lambda x: ' '.join(x.fillna('').astype(str)), axis=1))

            # Transform user features and movie features to TF-IDF vectors
            user_tfidf = self.tf_idf_vectorizer.transform(user_features_str)
            movie_tfidf = self.tf_idf_vectorizer.transform(self.imdb_dataset.apply(lambda x: ' '.join(x.fillna('').astype(str)), axis=1))

            # Calculate cosine similarity between user features and movie features
            similarity_scores = cosine_similarity(user_tfidf, movie_tfidf)

            # Get indices of top 5 similar movies
            top_indices = similarity_scores.argsort(axis=1)[:, ::-1][:, 1:6]

            # Print recommended movies
            print(f"Top 5 recommended movies for user {self.logged_in_user}:")
            for indices in top_indices[0][:5]:  # Only take the first 5 indices
                print(self.imdb_dataset.iloc[indices]['movie_title'])
        else:
            print("Please login first.")

    def initial_recommendation(self):
        # Concatenate all features into a single string for each movie in the IMDb dataset
        movie_features_str = self.imdb_dataset.apply(lambda x: ' '.join(x.fillna('').astype(str)), axis=1)

        # Fit TF-IDF vectorizer on movie features
        movie_tfidf = self.tf_idf_vectorizer.fit_transform(movie_features_str)

        # Calculate cosine similarity between all pairs of movies
        similarity_scores = cosine_similarity(movie_tfidf)

        # Get indices of top 5 similar movies for each movie
        top_indices = similarity_scores.argsort(axis=1)[:, ::-1][:, 1:6]
        
        print("Top 5 initial recommended movies:")
        for indices in top_indices[0][:5]:  # Only take the first 5 indices
            print(self.imdb_dataset.iloc[indices]['movie_title'])

    def record_user_history(self, movie_name):
        # Search for the movie in the IMDb dataset
        movie_row = self.imdb_dataset[self.imdb_dataset['movie_title'].str.contains(movie_name, case=False)]

        # Check if the movie exists
        if not movie_row.empty:
            # Extract all columns of the movie
            movie_info = movie_row.iloc[0].to_dict()

            movie_info['Username'] = username

            # Append the movie information to the user_history dataset
            self.user_history = pd.concat([self.user_history, pd.DataFrame(movie_info, index=[0])], ignore_index=True)
            # Save the updated user_history dataset back to the CSV file
            self.user_history.to_csv("User_history.csv", index=False, encoding='latin1')
           
            print(f"Movie '{movie_name}' recorded in user history.")
        else:
            print(f"Movie '{movie_name}' not found in the IMDb dataset.")

if __name__ == "__main__":
    system = MovieRecommendationSystem("movie_dataset.csv", "User_history.csv")
    username = input("Enter username: ")
    password = input("Enter password: ")
    if system.login(username, password, "authentication.txt"):
        system.recommend_movies()
        system.initial_recommendation()
        movie_title = input("Enter movie name: ")
        system.record_user_history(movie_title)
