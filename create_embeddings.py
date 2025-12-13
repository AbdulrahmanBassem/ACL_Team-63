import pandas as pd
import os
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Define the models we want to support
EMBEDDING_MODELS = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet": "sentence-transformers/all-mpnet-base-v2"
}

def create_feature_embeddings():
    # 1. Load Data
    reviews_path = 'Dataset/reviews.csv'
    hotels_path = 'Dataset/hotels.csv'
    
    if not os.path.exists(reviews_path) or not os.path.exists(hotels_path):
        print(f"Error: Files not found. Ensure {reviews_path} and {hotels_path} exist.")
        return

    print("Loading reviews and hotels...")
    reviews_df = pd.read_csv(reviews_path)
    hotels_df = pd.read_csv(hotels_path)

    # 2. Merge Dataframes to get Hotel Name
    print("Merging data...")
    df_merged = pd.merge(reviews_df, hotels_df[['hotel_id', 'hotel_name']], on='hotel_id', how='left')

    # 3. Select Columns and Handle NaNs
    feature_cols = [
        'hotel_id', 
        'hotel_name',
        'score_cleanliness', 
        'score_comfort', 
        'score_facilities', 
        'score_location', 
        'score_staff', 
        'score_value_for_money'
    ]
    
    df_features = df_merged[feature_cols].copy()
    df_features[feature_cols[2:]] = df_features[feature_cols[2:]].fillna(0)
    df_features['hotel_name'] = df_features['hotel_name'].fillna("Unknown Hotel")

    # 4. Create Semantic Feature Strings
    def build_feature_string(row):
        return (
            f"Hotel: {row['hotel_name']} (ID: {row['hotel_id']}). "
            f"Cleanliness: {row['score_cleanliness']}, "
            f"Comfort: {row['score_comfort']}, "
            f"Facilities: {row['score_facilities']}, "
            f"Location: {row['score_location']}, "
            f"Staff: {row['score_staff']}, "
            f"Value for Money: {row['score_value_for_money']}."
        )

    print("Constructing feature strings...")
    df_features['feature_text'] = df_features.apply(build_feature_string, axis=1)

    loader = DataFrameLoader(df_features, page_content_column="feature_text")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # ---------------------------------------------------------
    # LOOP THROUGH MODELS AND GENERATE INDICES
    # ---------------------------------------------------------
    for model_short_name, model_repo_id in EMBEDDING_MODELS.items():
        print(f"\n--- Processing Model: {model_short_name} ({model_repo_id}) ---")
        
        # Initialize Embedder
        embedding_model = HuggingFaceEmbeddings(model_name=model_repo_id)
        
        # Create Vector Store
        print(f"Generating embeddings for {model_short_name}...")
        vector_store = FAISS.from_documents(docs, embedding_model)

        # Save to a specific folder e.g., "feature_index_MiniLM"
        folder_name = f"feature_index_{model_short_name}"
        vector_store.save_local(folder_name)
        print(f"Success! Index saved to folder: '{folder_name}'")

if __name__ == "__main__":
    create_feature_embeddings()