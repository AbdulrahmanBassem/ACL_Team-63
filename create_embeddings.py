import pandas as pd
import os
import time
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from neo4j import GraphDatabase

EMBEDDING_MODELS = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet": "sentence-transformers/all-mpnet-base-v2"
}

BATCH_SIZE = 200 

def load_config(config_file='config.txt'):
    config = {}
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    config[key] = value
    return config

def create_feature_embeddings():
    config = load_config()
    NEO4J_URI = config.get('URI', 'neo4j://127.0.0.1:7687')
    NEO4J_USER = config.get('USERNAME', 'neo4j')
    NEO4J_PASSWORD = config.get('PASSWORD', 'password')

    reviews_path = 'Dataset/reviews.csv'
    hotels_path = 'Dataset/hotels.csv'
    
    if not os.path.exists(reviews_path) or not os.path.exists(hotels_path):
        print(f"Error: Files not found. Ensure {reviews_path} and {hotels_path} exist.")
        return

    print("Loading reviews and hotels...")
    reviews_df = pd.read_csv(reviews_path)
    hotels_df = pd.read_csv(hotels_path)

    print("Merging data...")
    df_merged = pd.merge(reviews_df, hotels_df[['hotel_id', 'hotel_name']], on='hotel_id', how='left')

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
    print(f"Total Documents to Process: {len(docs)}")

    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    for model_short_name, model_repo_id in EMBEDDING_MODELS.items():
        print(f"\n--- Processing Model: {model_short_name} ({model_repo_id}) ---")
        
        index_name = f"feature_index_{model_short_name}"
        node_label = f"Feature_{model_short_name}" 

       
        print(f"Cleaning existing '{node_label}' nodes in Neo4j...")
        with driver.session() as session:
            session.run(f"MATCH (n:{node_label}) DETACH DELETE n")
            try:
                session.run(f"DROP INDEX {index_name} IF EXISTS")
            except:
                pass

        embedding_model = HuggingFaceEmbeddings(model_name=model_repo_id)
        
        total_docs = len(docs)
        batches = [docs[i:i + BATCH_SIZE] for i in range(0, total_docs, BATCH_SIZE)]
        
        vector_store = None
        
        print(f"Starting ingestion in {len(batches)} batches of {BATCH_SIZE}...")
        
        for i, batch in enumerate(batches):
            print(f"  > Processing batch {i+1}/{len(batches)} ({len(batch)} docs)...")
            
            try:
                if vector_store is None:
                    # First batch: Create the Vector Store and Index
                    vector_store = Neo4jVector.from_documents(
                        batch,
                        embedding_model,
                        url=NEO4J_URI,
                        username=NEO4J_USER,
                        password=NEO4J_PASSWORD,
                        index_name=index_name,
                        node_label=node_label,
                        embedding_node_property="embedding",
                        text_node_property="text"
                    )
                else:
                    # Subsequent batches: Add to existing store
                    vector_store.add_documents(batch)
                
            except Exception as e:
                print(f"Error in batch {i+1}: {e}")
                # Optional: break or continue depending on desired behavior
                raise e

        print(f"Success! All embeddings for {model_short_name} saved to Neo4j.")

    driver.close()

if __name__ == "__main__":
    create_feature_embeddings()