import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def svm_classification(anime_csv, user_ratings_csv):

    # Load the datasets
    anime_df = pd.read_csv(anime_csv)
    user_ratings_df = pd.read_csv(user_ratings_csv)

    # Step 1: Feature Engineering

    # Average rating per user
    user_avg_rating = user_ratings_df.groupby('user_id')['rating'].mean().rename("user_avg_rating")
    user_features_df = user_ratings_df[['user_id']].drop_duplicates().merge(user_avg_rating, on='user_id')

    # Completion rate calculation
    completed_count = user_ratings_df[user_ratings_df['completion_status'] == 'Completed'].groupby('user_id').size()
    total_rated_count = user_ratings_df.groupby('user_id').size()
    completion_rate = (completed_count / total_rated_count).fillna(0).rename("completion_rate")
    user_features_df = user_features_df.merge(completion_rate, on='user_id', how='left')

    # Rating behavior (high and low rating frequency)
    high_rating_threshold = 8
    low_rating_threshold = 4
    high_rating_freq = user_ratings_df[user_ratings_df['rating'] >= high_rating_threshold].groupby('user_id').size() / total_rated_count
    low_rating_freq = user_ratings_df[user_ratings_df['rating'] <= low_rating_threshold].groupby('user_id').size() / total_rated_count
    high_rating_freq = high_rating_freq.rename("high_rating_freq").fillna(0)
    low_rating_freq = low_rating_freq.rename("low_rating_freq").fillna(0)
    user_features_df = user_features_df.merge(high_rating_freq, on='user_id', how='left')
    user_features_df = user_features_df.merge(low_rating_freq, on='user_id', how='left')

    # Step 2: Define the Rating Class Labels
    # Binary classification of ratings into "high" (1) and "low" (0) based on a threshold
    user_ratings_df['rating_class'] = user_ratings_df['rating'].apply(lambda x: 1 if x >= 7 else 0)

    # Merge user ratings with user features
    data = user_ratings_df.merge(user_features_df, on='user_id')
    # Step 3: Feature Selection and Splitting Data
    X = data[['user_avg_rating', 'completion_rate', 'high_rating_freq', 'low_rating_freq']]
    y = data['rating_class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Train the SVM Classifier
    svm_classifier = SVC(kernel='linear', random_state=42)
    svm_classifier.fit(X_train, y_train)

    # Step 5: Evaluate the Model
    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", report)
