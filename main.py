import csv
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommendationSystem:
    def __init__(self, imdb_dataset_path, user_history_path, user_searches_path):
        self.imdb_dataset = self.load_imdb_dataset(imdb_dataset_path)
        self.user_history = self.load_user_history(user_history_path)
        self.user_searches = self.load_user_searches(user_searches_path)
        self.logged_in_user = None
        self.tf_idf_vectorizer = TfidfVectorizer()
        self.search_array = []  # To store search inputs
        # self.tmdb_api_key = tmdb_api_key  # TMDB API key

    def load_imdb_dataset(self, path):
        return pd.read_csv(path)

    def load_user_history(self, path):
        return pd.read_csv(path, encoding='latin1')

    def load_user_searches(self, path):
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

            # Print recommended movies with all metadata
            print(f"Top 5 recommended movies for user {self.logged_in_user}:")
            for indices in top_indices[0][:5]:  # Only take the first 5 indices
                movie_metadata = self.imdb_dataset.iloc[indices]
                print(movie_metadata.to_dict())
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

            movie_info['Username'] = self.logged_in_user

            # Append the movie information to the user_history dataset
            self.user_history = pd.concat([self.user_history, pd.DataFrame(movie_info, index=[0])], ignore_index=True)
            # Save the updated user_history dataset back to the CSV file
            self.user_history.to_csv("User_history.csv", index=False, encoding='latin1')
           
            print(f"Movie '{movie_name}' recorded in user history.")
        else:
            print(f"Movie '{movie_name}' not found in the IMDb dataset.")
    
    def movie_category_filter(self, category):
        # Split the 'genres' column and check if the category is in the genres
        filtered_movies = self.imdb_dataset[self.imdb_dataset['genres'].str.contains(category, case=False, na=False)]

        # Get the top 100 movies that match the category
        top_100_movies = filtered_movies.head(100)

        # Return or print the metadata of the filtered movies
        print(f"Top 100 movies in category '{category}':")
        for _, movie in top_100_movies.iterrows():
            print(movie.to_dict())

    def on_change_search(self, search_input):
        if self.logged_in_user:
            # Append the search input to the search array
            self.search_array.append(search_input)

            # Search for the movie in the IMDb dataset
            movie_rows = self.imdb_dataset[self.imdb_dataset['movie_title'].str.contains(search_input, case=False, na=False)]

            # Check if any movies match the search input
            if not movie_rows.empty:
                # Create a copy of the movie_rows to avoid SettingWithCopyWarning
                movie_rows_copy = movie_rows.copy()
                
                # Add a column for the username
                movie_rows_copy['Username'] = self.logged_in_user

                # Append the search results to the user_searches dataset
                self.user_searches = pd.concat([self.user_searches, movie_rows_copy], ignore_index=True)
                
                # Save the updated user_searches dataset back to the CSV file
                self.user_searches.to_csv("User_searches.csv", index=False, encoding='latin1')
                
                print(f"Search for '{search_input}' recorded with matching movies.")
            else:
                print(f"No movies found for search input '{search_input}'.")
        else:
            print("Please login first.")

    def get_movie_poster(self, movie_id):
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=9eeecccb8c47f32eeae0f068e5f6348f"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'poster_path' in data and data['poster_path']:
                image_url = f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
                return image_url
            else:
                print(f"No poster found for movie ID '{movie_id}'")
                return None
        else:
            print(f"Failed to retrieve data from TMDB: {response.status_code}")
            return None

if __name__ == "__main__":
    # tmdb_api_key = 'YOUR_TMDB_API_KEY'  # Replace with your TMDB API key
    system = MovieRecommendationSystem("movie_dataset.csv", "User_history.csv", "User_searches.csv")
    username = input("Enter username: ")
    password = input("Enter password: ")
    if system.login(username, password, "authentication.txt"):
        system.recommend_movies()
        system.initial_recommendation()
        category = input("Enter movie category: ")
        system.movie_category_filter(category)
        movie_title = input("Enter movie name: ")
        system.record_user_history(movie_title)
        search_input = input("Search for a movie: ")
        system.on_change_search(search_input)
        movie_id = input("Enter movie ID: ")
        image_url = system.get_movie_poster(movie_id)
        if image_url:
            print(f"Image URL for movie ID '{movie_id}': {image_url}")
