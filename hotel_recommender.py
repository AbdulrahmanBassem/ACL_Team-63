import os
import pandas as pd
from neo4j import GraphDatabase
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# --- NEW IMPORTS FOR THE WRAPPER (From Lab 8) ---
from huggingface_hub import InferenceClient
from langchain_core.language_models.llms import LLM
from typing import Optional, List, Any
from pydantic import Field
# ------------------------------------------------

# 1. CONFIGURATION & SETUP
def load_config(config_file='config.txt'):
    config = {}
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    config[key] = value
    return config

neo4j_config = load_config()
NEO4J_URI = neo4j_config.get('URI', 'neo4j://127.0.0.1:7687')
NEO4J_USER = neo4j_config.get('USERNAME', 'neo4j')
NEO4J_PASSWORD = neo4j_config.get('PASSWORD', 'password')

# REPLACE THIS WITH YOUR TOKEN
HF_TOKEN = "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" 

# 2. VECTOR STORE SETUP
print("Loading Reviews for Vector Store...")
reviews_df = pd.read_csv('Dataset/reviews.csv')
hotels_df = pd.read_csv('Dataset/hotels.csv')
df_merged = pd.merge(reviews_df, hotels_df[['hotel_id', 'hotel_name']], on='hotel_id', how='left')
df_merged['combined_text'] = "Hotel: " + df_merged['hotel_name'].astype(str) + ". Review: " + df_merged['review_text'].astype(str)

sample_df = df_merged.sample(n=1000, random_state=42)
loader = DataFrameLoader(sample_df, page_content_column="combined_text")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

print("Creating Embeddings...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Building FAISS Index...")
vector_store = FAISS.from_documents(docs, embedding_model)
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# 3. NEO4J GRAPH RETRIEVER
class HotelGraphRetriever:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_hotels_by_city(self, city_name):
        query = """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)
        WHERE toLower(c.name) CONTAINS toLower($city)
        RETURN h.name AS Hotel, h.average_reviews_score AS Score
        ORDER BY h.average_reviews_score DESC LIMIT 5
        """
        with self.driver.session() as session:
            result = session.run(query, city=city_name)
            return [f"Hotel: {record['Hotel']} (Rating: {record['Score']})" for record in result]

graph_db = HotelGraphRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# 4. HYBRID RAG SYSTEM
def hybrid_retrieval(user_query, city_extraction=None):
    context_parts = []
    if city_extraction:
        print(f"--> [Graph] Fetching top hotels in {city_extraction}...")
        graph_results = graph_db.get_hotels_by_city(city_extraction)
        if graph_results:
            context_parts.append(f"Top Rated Hotels in {city_extraction} from Knowledge Graph:\n" + "\n".join(graph_results))
    
    print("--> [Vector] Searching reviews for semantic match...")
    vector_results = vector_retriever.invoke(user_query)
    vector_text = "\n".join([doc.page_content for doc in vector_results])
    context_parts.append(f"Relevant User Reviews:\n{vector_text}")
    
    return "\n\n".join(context_parts)

# ---------------------------------------------------------
# 5. LLM GENERATION - FIXED WITH WRAPPER (Lab 8 Method)
# ---------------------------------------------------------

# Define the Wrapper Class from Lab 8 [cite: 305]
class GemmaLangChainWrapper(LLM):
    client: Any = Field(...)
    max_tokens: int = 500
    
    @property
    def _llm_type(self) -> str:
        return "gemma_hf_api"
        
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Uses chat_completion which is supported by Gemma-IT
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.2
        )
        return response.choices[0].message["content"]

# Initialize the Client
client = InferenceClient(
    model="google/gemma-2-2b-it", 
    token=HF_TOKEN
)

# Initialize the Custom LLM
llm = GemmaLangChainWrapper(client=client)

template = """
You are a helpful Hotel Recommender Assistant. 
Use the following context (Graph Data and User Reviews) to answer the user's request.
If you recommend a hotel, explain why based on the reviews or ratings provided.

Context:
{context}

User Question: {question}

Answer:
"""

prompt_template = PromptTemplate.from_template(template)

def recommend_hotel(user_query, city=None):
    # 1. Retrieve Context
    context = hybrid_retrieval(user_query, city)
    
    # 2. Generate Answer
    prompt = prompt_template.format(context=context, question=user_query)
    print("--> [LLM] Generating response...")
    
    # Invoke the custom wrapper
    response = llm.invoke(prompt)
    
    return response

# 6. EXECUTION EXAMPLE
if __name__ == "__main__":
    user_q = "I am looking for a hotel with good cleanliness."
    city_entity = "Paris"
    
    print(f"\nUser Query: {user_q} (City detected: {city_entity})")
    print("-" * 50)
    
    try:
        answer = recommend_hotel(user_q, city=city_entity)
        print("\n=== AI Recommendation ===\n")
        print(answer)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        graph_db.close()