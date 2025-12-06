import streamlit as st
import pandas as pd
import os
from neo4j import GraphDatabase
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from huggingface_hub import InferenceClient
from langchain_core.language_models.llms import LLM
from typing import Optional, List, Any
from pydantic import Field

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Hotel Graph-RAG Assistant", layout="wide")

# LOAD CONFIG
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

# --- USER TOKEN INPUT ---
# In a real app, use st.secrets. For now, we use a sidebar input or hardcode.
with st.sidebar:
    st.header("Configuration")
    hf_token = st.text_input("Enter Hugging Face Token:", type="password")
    st.info("Required to access Gemma-2b-it")

# ---------------------------------------------------------
# 2. CUSTOM GEMMA WRAPPER (From Lab 8)
# ---------------------------------------------------------
class GemmaLangChainWrapper(LLM):
    client: Any = Field(...)
    max_tokens: int = 500
    
    @property
    def _llm_type(self) -> str:
        return "gemma_hf_api"
        
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.2
        )
        return response.choices[0].message["content"]

# ---------------------------------------------------------
# 3. INITIALIZE RESOURCES (Cached)
# ---------------------------------------------------------

@st.cache_resource
def setup_vector_store():
    """Loads CSV, creates embeddings and FAISS index. Cached for performance."""
    st.write("Creating Vector Store (This happens only once)...")
    
    # Load Data
    reviews_df = pd.read_csv('Dataset/reviews.csv')
    hotels_df = pd.read_csv('Dataset/hotels.csv')
    
    # Merge
    df_merged = pd.merge(reviews_df, hotels_df[['hotel_id', 'hotel_name']], on='hotel_id', how='left')
    df_merged['combined_text'] = "Hotel: " + df_merged['hotel_name'].astype(str) + ". Review: " + df_merged['review_text'].astype(str)
    
    # Sample for speed (Adjust as needed)
    sample_df = df_merged.sample(n=1000, random_state=42)
    
    # Loader
    loader = DataFrameLoader(sample_df, page_content_column="combined_text")
    documents = loader.load()
    
    # Splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    # Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Vector Store
    vector_store = FAISS.from_documents(docs, embedding_model)
    return vector_store.as_retriever(search_kwargs={"k": 4})

@st.cache_resource
def setup_graph_db():
    """Connects to Neo4j. Cached."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ---------------------------------------------------------
# 4. RETRIEVAL LOGIC
# ---------------------------------------------------------
def get_hotels_by_city(driver, city_name):
    query = """
    MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)
    WHERE toLower(c.name) CONTAINS toLower($city)
    RETURN h.name AS Hotel, h.average_reviews_score AS Score
    ORDER BY h.average_reviews_score DESC LIMIT 5
    """
    with driver.session() as session:
        result = session.run(query, city=city_name)
        return [f"Hotel: {record['Hotel']} (Rating: {record['Score']})" for record in result]

def generate_response(user_query, city, vector_retriever, driver, llm_client):
    # 1. Hybrid Retrieval
    context_parts = []
    
    # Graph Layer (Structured)
    if city:
        graph_results = get_hotels_by_city(driver, city)
        if graph_results:
            context_parts.append(f"Top Rated Hotels in {city} (from Knowledge Graph):\n" + "\n".join(graph_results))
    
    # Vector Layer (Unstructured)
    vector_results = vector_retriever.invoke(user_query)
    vector_text = "\n".join([doc.page_content for doc in vector_results])
    context_parts.append(f"Relevant User Reviews (from Vector Store):\n{vector_text}")
    
    full_context = "\n\n".join(context_parts)
    
    # 2. LLM Generation
    template = """
    You are a helpful Hotel Recommender Assistant. 
    Use the following context (Graph Data and User Reviews) to answer the user's request.
    If you recommend a hotel, explain why based on the reviews or ratings provided.

    Context:
    {context}

    User Question: {question}

    Answer:
    """
    prompt = PromptTemplate.from_template(template).format(context=full_context, question=user_query)
    
    # Init Wrapper
    llm = GemmaLangChainWrapper(client=llm_client)
    response = llm.invoke(prompt)
    
    return response, full_context

# ---------------------------------------------------------
# 5. MAIN UI
# ---------------------------------------------------------
st.title("🏨 Graph-RAG Hotel Assistant")
st.markdown("Combines **Neo4j** (Facts) and **FAISS** (Reviews) to recommend the perfect stay.")

# Inputs
col1, col2 = st.columns([3, 1])
with col1:
    user_query = st.text_input("What are you looking for?", placeholder="e.g., A romantic hotel with good breakfast")
with col2:
    city_entity = st.text_input("City (Entity)", placeholder="e.g., Amsterdam")

if st.button("Ask Assistant"):
    if not hf_token:
        st.error("Please enter your Hugging Face Token in the sidebar!")
    else:
        try:
            with st.spinner("Initializing AI Components..."):
                # Load Resources
                vector_retriever = setup_vector_store()
                driver = setup_graph_db()
                llm_client = InferenceClient(model="google/gemma-2-2b-it", token=hf_token)
            
            with st.spinner("Thinking..."):
                answer, context = generate_response(user_query, city_entity, vector_retriever, driver, llm_client)
            
            # Display Answer
            st.success("Recommendation:")
            st.write(answer)
            
            # Display Debugging/Context (Requirement: Transparency)
            with st.expander("🔍 See Retrieved Context (Graph + Vector)"):
                st.text(context)
                
        except Exception as e:
            st.error(f"An error occurred: {e}")