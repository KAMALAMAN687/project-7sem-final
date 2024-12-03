import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download necessary NLTK resources
nltk.download('stopwords')

# Load the CSV file
@st.cache_data
def load_data():
    return pd.read_csv("product_reviews.csv")

# Sentiment analysis function (manual labeling or fallback)
def analyze_sentiment(review):
    try:
        if not isinstance(review, str):
            review = str(review)  # Convert non-strings to strings
        
        sentiment = TextBlob(review).sentiment.polarity
        if sentiment > 0.1:
            return 'positive'
        elif sentiment > 0.05:
            return 'neutral'
        else:
            return 'negative'
    except Exception as e:
        print(f"Error processing review: {review}. Error: {e}")
        return 'neutral'

# Apply sentiment analysis to each review
def preprocess_data(data):
    data['reviews'] = data['reviews'].fillna('')  # Handle NaN values
    data['true_sentiment'] = data['reviews'].apply(analyze_sentiment)
    data['reviews_clean'] = data['reviews'].str.lower()  # Clean the review text
    return data

# Prepare training and testing data
def prepare_data(data):
    X = data['reviews_clean']
    y = data['true_sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# Vectorize text data using TF-IDF
def vectorize_data(X_train, X_test):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer

# Train RandomForestClassifier for better accuracy
def train_model(X_train_vec, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_vec, y_train)
    return model

# Evaluate model performance
def evaluate_model(model, X_test_vec, y_test):
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=['positive', 'neutral', 'negative'])
    return accuracy, precision, recall, f1, conf_matrix

# Display results in Streamlit
def display_results(accuracy, precision, recall, f1, conf_matrix, top_products_per_category):
    st.subheader("Model Performance Metrics:")
    st.write(f"Accuracy: {accuracy * 100:.2f}%")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1-Score: {f1:.2f}")
    
    st.subheader("Confusion Matrix:")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['positive', 'neutral', 'negative'], yticklabels=['positive', 'neutral', 'negative'])
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    st.pyplot(fig)
    
    st.subheader("Top 5 Products in Each Category by Sentiment Score")
    st.write(top_products_per_category)

# Function for chatbot interaction
def chatbot_interface(data, top_products_per_category):
    st.header("Product Review Analysis Chatbot")
    user_input = st.text_input("Enter a product category or keyword (e.g., 'laptops', 'great battery life'):")

    if user_input:
        # Match category or keyword
        category_matches = [cat for cat in data['categories'].unique() if cat.lower() in user_input.lower()]
        keyword_matches = data[data['reviews_clean'].str.contains(user_input.lower())]
        
        if category_matches:
            category = category_matches[0]
            top_products = top_products_per_category[top_products_per_category['categories'] == category]
            if not top_products.empty:
                st.write(f"### Top Products in '{category}' category:")
                st.write(top_products[['name', 'sentiment_score']])
            else:
                st.write(f"No top products found in the '{category}' category.")
        elif not keyword_matches.empty:
            st.write(f"### Products matching the keyword '{user_input}':")
            st.write(keyword_matches[['name', 'reviews', 'true_sentiment']])
        else:
            st.write("No matching products or categories found. Try again!")

# Simple recommendation system
def recommend_products(data, selected_product):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['reviews_clean'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    product_index = data[data['name'] == selected_product].index[0]
    similar_indices = cosine_sim[product_index].argsort()[-6:-1][::-1]
    recommendations = data.iloc[similar_indices][['name', 'reviews']]
    return recommendations

# Main function
def main():
    # Load and preprocess data
    data = load_data()
    data = preprocess_data(data)
    
    # Prepare data for training/testing
    X_train, X_test, y_train, y_test = prepare_data(data)
    X_train_vec, X_test_vec, vectorizer = vectorize_data(X_train, X_test)
    
    # Train and evaluate model
    model = train_model(X_train_vec, y_train)
    accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, X_test_vec, y_test)
    
    # Calculate top 5 products per category
    data['sentiment_score'] = data['reviews'].apply(lambda x: TextBlob(x).sentiment.polarity)
    product_avg_sentiment = data.groupby(['categories', 'name']).agg({'sentiment_score': 'mean'}).reset_index()
    top_products_per_category = product_avg_sentiment.groupby('categories').apply(lambda x: x.nlargest(5, 'sentiment_score')).reset_index(drop=True)
    
    # Display results
    display_results(accuracy, precision, recall, f1, conf_matrix, top_products_per_category)
    
    # Chatbot interaction
    chatbot_interface(data, top_products_per_category)
    
    # Recommendation system
    selected_product = st.selectbox("Select a product to get recommendations", data['name'].unique())
    if selected_product:
        recommendations = recommend_products(data, selected_product)
        st.write("### Recommended Products:")
        st.write(recommendations)

# Run the app
if __name__ == "__main__":
    main()
