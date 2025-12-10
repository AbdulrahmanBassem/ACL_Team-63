import pandas as pd
import os
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_feature_embeddings():
    # 1. Load Data
    csv_path = 'Dataset/reviews.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print("Loading reviews...")
    df = pd.read_csv(csv_path)

    # 2. Select Columns and Handle NaNs
    feature_cols = [
        'hotel_id', 
        'score_cleanliness', 
        'score_comfort', 
        'score_facilities', 
        'score_location', 
        'score_staff', 
        'score_value_for_money'
    ]
    
    # Fill missing scores with 0 or a neutral value
    df_features = df[feature_cols].copy().fillna(0)

    # 3. Create Semantic Feature Strings
    # This converts the numerical vector into a text description the LLM/Embedder can understand.
    # Format: "Hotel ID: <id>. Cleanliness: <score>, Comfort: <score>, ..."
    def build_feature_string(row):
        return (
            f"Hotel ID: {row['hotel_id']}. "
            f"Cleanliness: {row['score_cleanliness']}, "
            f"Comfort: {row['score_comfort']}, "
            f"Facilities: {row['score_facilities']}, "
            f"Location: {row['score_location']}, "
            f"Staff: {row['score_staff']}, "
            f"Value for Money: {row['score_value_for_money']}."
        )

    print("Constructing feature strings...")
    df_features['feature_text'] = df_features.apply(build_feature_string, axis=1)

    # 4. Prepare for Vector Store
    # We use LangChain's loader to treat these strings as "documents"
    loader = DataFrameLoader(df_features, page_content_column="feature_text")
    documents = loader.load()

    # 5. Splitter (Optional for single sentences, but good practice for consistency)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # 6. Generate Embeddings & Build Index
    print("Generating embeddings (this may take a moment)...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vector_store = FAISS.from_documents(docs, embedding_model)

    # 7. Save Index Locally
    index_name = "feature_vector_index"
    vector_store.save_local(index_name)
    print(f"Success! FAISS index saved to folder: '{index_name}'")

if __name__ == "__main__":
    create_feature_embeddings()