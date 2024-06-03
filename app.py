import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load the saved model and preprocessors
with open('/workspaces/final_dsssss/laptop_recommender.pkl2', 'rb') as f:
    df, tfidf, tfidf_matrix, cosine_sim = pickle.load(f)

# Convert the 'new price' column to numeric for filtering
df['new price'] = df['new price'].astype(str).str.replace(r'[^\d.]', '', regex=True).astype(float)
# Get unique options for suitable_for and os_system
suitable_for_options = df['suitable for'].unique()
os_system_options = df['os system'].unique()

# Function to get recommendations
def get_recommendations(suitable_for, os_system, budget, top_n=20):
    filtered_data = df[
        df['suitable for'].str.contains(suitable_for, case=False, na=False) &
        df['os system'].str.contains(os_system, case=False, na=False) &
        (df['new price'] <= budget)
    ]
    if filtered_data.empty:
        return []
    filtered_indices = filtered_data.index.tolist()

    # Chỉ sử dụng cosine similarity trên các hàng đã được lọc
    sim_scores = cosine_sim[filtered_indices, :][:, filtered_indices]

    mean_sim_scores = sim_scores.mean(axis=0)
    similarity_scores = list(enumerate(mean_sim_scores))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_products = [filtered_indices[i[0]] for i in similarity_scores[:top_n]]  # Lấy chỉ số từ filtered_indices
    recommended_products = df.iloc[top_products][['name', 'description', 'picture-src']].copy()
    return recommended_products.to_dict('records')

# Streamlit app
st.title("Laptop Recommender")

suitable_for = st.selectbox("Demand", suitable_for_options)
os_system = st.selectbox("Favorite", os_system_options)
budget = st.number_input("Budget", min_value=0)

if st.button("Get Recommendations"):
    recommendations = get_recommendations(suitable_for, os_system, budget, top_n=5)
    if recommendations:
        for rec in recommendations:
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; border: 1px solid #e6e6e6; padding: 10px; margin-bottom: 10px; border-radius: 10px;">
                    <img src="{rec['picture-src']}" alt="{rec['name']}" style="width: 150px; height: auto; margin-right: 20px; border-radius: 10px;">
                    <div>
                        <h3>{rec['name']}</h3>
                        <p>{rec['description']}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.write("No suitable laptops found.")
