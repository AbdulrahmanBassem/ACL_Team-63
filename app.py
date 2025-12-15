import streamlit as st
import pandas as pd
import os
import re
import graphviz  
from neo4j import GraphDatabase
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Neo4jVector
from langchain_core.prompts import PromptTemplate
from huggingface_hub import InferenceClient
from langchain_core.language_models.llms import LLM
from typing import Optional, List, Any
from pydantic import Field

st.set_page_config(page_title="Hotel Graph-RAG Assistant", layout="wide")

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
HF_TOKEN = neo4j_config.get('HFToken', '')

EMBEDDING_MODELS = {
    "MiniLM (Fast)": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet (Accurate)": "sentence-transformers/all-mpnet-base-v2"
}

LLM_MODELS = {
    "Gemma (2B)": "google/gemma-2-2b-it",
    "Mistral (7B - v0.2)": "mistralai/Mistral-7B-Instruct-v0.2",
    "Llama 3 (8B)": "meta-llama/Meta-Llama-3-8B-Instruct"
}


def map_range_to_groups(user_min, user_max):
    dataset_groups = {
        "18-24": (18, 24),
        "25-34": (25, 34),
        "35-44": (35, 44),
        "45-54": (45, 54),
        "55+":   (55, 100)
    }
    selected_groups = []
    for label, (g_min, g_max) in dataset_groups.items():
        overlap_start = max(user_min, g_min)
        overlap_end = min(user_max, g_max)
        if overlap_start <= overlap_end:
            selected_groups.append(label)
    return selected_groups


class HuggingFaceLLMWrapper(LLM):
    """
    Generic Wrapper for Hugging Face Inference API.
    """
    client: Any = Field(...)
    model_name: str = Field(...)
    max_tokens: int = 500
    
    @property
    def _llm_type(self) -> str:
        return "hf_inference_api"
        
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.2
        )
        return response.choices[0].message["content"]

@st.cache_resource
def setup_text_vector_store(model_repo_id):
    reviews_df = pd.read_csv('Dataset/reviews.csv')
    hotels_df = pd.read_csv('Dataset/hotels.csv')
    df_merged = pd.merge(reviews_df, hotels_df[['hotel_id', 'hotel_name']], on='hotel_id', how='left')
    df_merged['combined_text'] = "Hotel: " + df_merged['hotel_name'].astype(str) + ". Review: " + df_merged['review_text'].astype(str)
    
    sample_df = df_merged.sample(n=1000, random_state=42)
    
    loader = DataFrameLoader(sample_df, page_content_column="combined_text")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    embedding_model = HuggingFaceEmbeddings(model_name=model_repo_id)
    vector_store = FAISS.from_documents(docs, embedding_model)
    return vector_store.as_retriever(search_kwargs={"k": 3})

@st.cache_resource
def setup_feature_vector_store_neo4j(model_short_key, model_repo_id):
    clean_key = model_short_key.split(" ")[0] 
    index_name = f"feature_index_{clean_key}"
    embedding_model = HuggingFaceEmbeddings(model_name=model_repo_id)
    try:
        vector_store = Neo4jVector.from_existing_index(
            embedding_model,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            index_name=index_name,
            embedding_node_property="embedding",
            text_node_property="text"
        )
        return vector_store.as_retriever(search_kwargs={"k": 10}) # Fetch more initially, then deduplicate
    except Exception as e:
        st.error(f"Failed to connect to Neo4j Index '{index_name}': {e}")
        return None

@st.cache_resource
def setup_graph_db():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

@st.cache_resource
def get_all_entities(_driver):
    entities = {
        "City": set(), "Country": set(), "Hotel": set(), 
        "Traveller Type": set(), "Gender": set(), "Age Group": set()
    }
    with _driver.session() as session:
        result = session.run("MATCH (c:City) RETURN c.name AS val")
        entities["City"] = {r["val"] for r in result if r["val"]}
        result = session.run("MATCH (c:Country) RETURN c.name AS val")
        entities["Country"] = {r["val"] for r in result if r["val"]}
        result = session.run("MATCH (h:Hotel) RETURN h.name AS val")
        entities["Hotel"] = {r["val"] for r in result if r["val"]}
        result = session.run("MATCH (t:Traveller) RETURN DISTINCT t.type AS val")
        entities["Traveller Type"] = {r["val"] for r in result if r["val"]}
        result = session.run("MATCH (t:Traveller) RETURN DISTINCT t.gender AS val")
        entities["Gender"] = {r["val"] for r in result if r["val"]}
        result = session.run("MATCH (t:Traveller) RETURN DISTINCT t.age AS val")
        entities["Age Group"] = {r["val"] for r in result if r["val"]}
    return entities


# Intent classification and entity extraction
def extract_entities_from_query(query, entity_db):
    query_lower = query.lower()
    detected = {}
    
    for category, values in entity_db.items():
        if category == "Age Group": continue
        found_val = None
        longest_len = 0
        for val in values:
            if str(val).lower() in query_lower:
                if len(str(val)) > longest_len:
                    found_val = val
                    longest_len = len(str(val))
        if found_val:
            detected[category] = found_val

    
    age_pattern = r"(?:between\s+)?(\d+)\s*(?:to|and|-)\s*(\d+)"
    age_match = re.search(age_pattern, query_lower)
    if age_match:
        min_age = int(age_match.group(1))
        max_age = int(age_match.group(2))
        relevant_groups = map_range_to_groups(min_age, max_age)
        if relevant_groups:
            detected["Target Age Groups"] = relevant_groups

    if any(k in query_lower for k in ["cleanliness", "clean", "hygiene", "tidy"]): detected["Sort By"] = "Cleanliness"
    if any(k in query_lower for k in ["value", "money", "worth", "price"]): detected["Sort By"] = "Value"
    if any(k in query_lower for k in ["location", "central", "convenient"]): detected["Sort By"] = "Location"
    if any(k in query_lower for k in ["comfort", "comfortable", "cozy"]): detected["Sort By"] = "Comfort"

    return detected

#Graph Visualization
def get_visualization_data(driver, intent, entities):
    """
    Returns a list of tuples (source, relationship, target) for graph visualization.
    """
    edges = []
    with driver.session() as session:
        if intent == "City" and "City" in entities:
            query = """
            MATCH (h:Hotel)-[r:LOCATED_IN]->(c:City)
            WHERE toLower(c.name) = toLower($city)
            RETURN h.name AS source, 'LOCATED_IN' AS rel, c.name AS target LIMIT 5
            """
            result = session.run(query, city=entities["City"])
            edges = [(r["source"], r["rel"], r["target"]) for r in result]

        elif intent == "Country" and "Country" in entities:
            query = """
            MATCH (h:Hotel)-[:LOCATED_IN]->(:City)-[:LOCATED_IN]->(c:Country)
            WHERE toLower(c.name) = toLower($country)
            RETURN h.name AS source, 'LOCATED_IN' AS rel, c.name AS target LIMIT 5
            """
            result = session.run(query, country=entities["Country"])
            edges = [(r["source"], r["rel"], r["target"]) for r in result]

        elif intent == "Traveller Type" and "Traveller Type" in entities:
            query = """
            MATCH (t:Traveller)-[:WROTE]->(:Review)-[:REVIEWED]->(h:Hotel)
            WHERE toLower(t.type) = toLower($type)
            RETURN t.type + ' Traveller' AS source, 'LIKES' AS rel, h.name AS target LIMIT 5
            """
            result = session.run(query, type=entities["Traveller Type"])
            edges = [(r["source"], r["rel"], r["target"]) for r in result]

        elif intent == "Gender" and "Gender" in entities:
            query = """
            MATCH (t:Traveller)-[:WROTE]->(:Review)-[:REVIEWED]->(h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(cntry:Country)
            WHERE toLower(t.gender) = toLower($gender)
            RETURN t.gender + ' Traveller' AS source, 'VISITED' AS rel, cntry.name AS target LIMIT 5
            """
            result = session.run(query, gender=entities["Gender"])
            edges = [(r["source"], r["rel"], r["target"]) for r in result]
            
        elif intent == "Age Group" and "Target Age Groups" in entities:
            group = entities["Target Age Groups"][0]
            query = """
            MATCH (t:Traveller)-[:WROTE]->(:Review)-[:REVIEWED]->(h:Hotel)
            WHERE t.age = $group
            RETURN 'Age Group ' + $group AS source, 'RATED_HIGHLY' AS rel, h.name AS target LIMIT 5
            """
            result = session.run(query, group=group)
            edges = [(r["source"], r["rel"], r["target"]) for r in result]

        elif intent == "Sort By" and "Sort By" in entities:
            sort_key = entities["Sort By"]
            if "City" in entities:
                query = """
                MATCH (h:Hotel)-[r:LOCATED_IN]->(c:City)
                WHERE toLower(c.name) = toLower($city)
                RETURN c.name AS source, 'CONTAINS_TOP_HOTEL' AS rel, h.name AS target LIMIT 5
                """
                result = session.run(query, city=entities["City"])
                edges = [(r["source"], r["rel"], r["target"]) for r in result]
            elif "Country" in entities:
                query = """
                MATCH (h:Hotel)-[:LOCATED_IN]->(:City)-[:LOCATED_IN]->(c:Country)
                WHERE toLower(c.name) = toLower($country)
                RETURN c.name AS source, 'CONTAINS_TOP_HOTEL' AS rel, h.name AS target LIMIT 5
                """
                result = session.run(query, country=entities["Country"])
                edges = [(r["source"], r["rel"], r["target"]) for r in result]
            else:
                prop_map = {
                    "Cleanliness": "score_cleanliness",
                    "Value": "score_value_for_money",
                    "Location": "score_location",
                    "Comfort": "score_comfort"
                }
                db_prop = prop_map.get(sort_key, "average_reviews_score")
                
                query = f"""
                MATCH (h:Hotel)
                WHERE h.{db_prop} IS NOT NULL
                RETURN 'Global Top' AS source, 'BEST_{sort_key.upper()}' AS rel, h.name AS target
                ORDER BY h.{db_prop} DESC LIMIT 5
                """
                result = session.run(query)
                edges = [(r["source"], r["rel"], r["target"]) for r in result]

        elif "City" in entities:
            query = """
            MATCH (h:Hotel)-[r:LOCATED_IN]->(c:City)
            WHERE toLower(c.name) = toLower($city)
            RETURN h.name AS source, 'LOCATED_IN' AS rel, c.name AS target LIMIT 5
            """
            result = session.run(query, city=entities["City"])
            edges = [(r["source"], r["rel"], r["target"]) for r in result]
        elif "Country" in entities:
            query = """
            MATCH (h:Hotel)-[:LOCATED_IN]->(:City)-[:LOCATED_IN]->(c:Country)
            WHERE toLower(c.name) = toLower($country)
            RETURN c.name AS source, 'LOCATED_IN' AS rel, h.name AS target LIMIT 5
            """
            result = session.run(query, country=entities["Country"])
            edges = [(r["source"], r["rel"], r["target"]) for r in result]
            
    return edges

#basline 
def get_hotels_by_city(driver, city_name):
    query = """
    MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)
    WHERE toLower(c.name) = toLower($city)
    RETURN h.name AS Hotel, h.average_reviews_score AS Score
    ORDER BY h.average_reviews_score DESC LIMIT 5
    """
    with driver.session() as session:
        result = session.run(query, city=city_name)
        return [f"Hotel: {record['Hotel']} (Rating: {record['Score']})" for record in result], query

def get_hotels_by_country(driver, country_name):
    query = """
    MATCH (h:Hotel)-[:LOCATED_IN]->(:City)-[:LOCATED_IN]->(c:Country)
    WHERE toLower(c.name) = toLower($country)
    RETURN h.name AS Hotel, h.average_reviews_score AS Score
    ORDER BY h.average_reviews_score DESC LIMIT 5
    """
    with driver.session() as session:
        result = session.run(query, country=country_name)
        return [f"Hotel: {record['Hotel']} (Rating: {record['Score']})" for record in result], query

def get_hotels_by_traveller_type(driver, traveller_type):
    query = """
    MATCH (t:Traveller)-[:WROTE]->(r:Review)-[:REVIEWED]->(h:Hotel)
    WHERE toLower(t.type) = toLower($type)
    WITH h.name AS hotelName, AVG(r.score_overall) AS avgRating
    ORDER BY avgRating DESC LIMIT 3
    RETURN hotelName, avgRating
    """
    with driver.session() as session:
        result = session.run(query, type=traveller_type)
        return [f"Hotel: {record['hotelName']} (Avg Rating: {round(record['avgRating'], 2)})" for record in result], query

def get_best_countries_by_traveller_type(driver, traveller_type):
    query = """
    MATCH (t:Traveller)-[:WROTE]->(r:Review)-[:REVIEWED]->(h:Hotel)-->(:City)-->(cntry:Country)
    WHERE toLower(t.type) = toLower($type)
    WITH cntry.name AS Country, AVG(r.score_overall) AS avgRating
    ORDER BY avgRating DESC LIMIT 3
    RETURN Country, avgRating
    """
    with driver.session() as session:
        result = session.run(query, type=traveller_type)
        return [f"Country: {record['Country']} (Avg Rating: {round(record['avgRating'], 2)})" for record in result], query

def get_top_countries_by_gender(driver, gender):
    query = """
    MATCH (t:Traveller)-[:WROTE]->(:Review)-[:REVIEWED]->(h:Hotel)-->(:City)-->(cntry:Country)
    WHERE toLower(t.gender) = toLower($gender)
    RETURN cntry.name AS Country, count(*) AS Visits
    ORDER BY Visits DESC LIMIT 3
    """
    with driver.session() as session:
        result = session.run(query, gender=gender)
        return [f"Country: {record['Country']} ({record['Visits']} visits)" for record in result], query

def get_top_countries_by_age_groups(driver, group_list):
    query = """
    MATCH (t:Traveller)-[:WROTE]->(:Review)-[:REVIEWED]->(h:Hotel)-->(:City)-->(cntry:Country)
    WHERE t.age IN $groups
    RETURN cntry.name AS Country, count(*) AS Visits
    ORDER BY Visits DESC LIMIT 3
    """
    with driver.session() as session:
        result = session.run(query, groups=group_list)
        return [f"Country: {record['Country']} ({record['Visits']} visits)" for record in result], query

def get_top_hotels_by_age_groups(driver, group_list):
    query = """
    MATCH (t:Traveller)-[:WROTE]->(r:Review)-[:REVIEWED]->(h:Hotel)
    WHERE t.age IN $groups
    WITH h.name AS Hotel, AVG(r.score_overall) AS avgRating
    ORDER BY avgRating DESC LIMIT 3
    RETURN Hotel, avgRating
    """
    with driver.session() as session:
        result = session.run(query, groups=group_list)
        return [f"Hotel: {record['Hotel']} (Avg Rating: {round(record['avgRating'], 2)})" for record in result], query

def get_best_hotels_by_cleanliness(driver, city=None, country=None):
    query_base = """
    MATCH (r:Review)-[:REVIEWED]->(h:Hotel) 
    WHERE r.score_cleanliness IS NOT NULL
    """
    params = {}
    if city:
        query_base += """
        MATCH (h)-[:LOCATED_IN]->(c:City) 
        WHERE toLower(c.name) = toLower($city) 
        """
        params["city"] = city
    elif country:
        query_base += """
        MATCH (h)-[:LOCATED_IN]->(:City)-->(cntry:Country) 
        WHERE toLower(cntry.name) = toLower($country) 
        """
        params["country"] = country
    query_end = """
    WITH h.name AS Hotel, AVG(r.score_cleanliness) AS AvgScore 
    RETURN Hotel, AvgScore 
    ORDER BY AvgScore DESC 
    LIMIT 5
    """
    with driver.session() as session:
        result = session.run(query_base + query_end, **params)
        data = [f"Hotel: {record['Hotel']} (Cleanliness: {round(record['AvgScore'], 2)})" for record in result]
        scope = f"in {city}" if city else (f"in {country}" if country else "Globally")
        return (data if data else [f"No cleanliness data found {scope}."]), query_base + query_end

def get_best_hotels_by_value(driver, city=None, country=None):
    query_base = """
    MATCH (r:Review)-[:REVIEWED]->(h:Hotel) 
    WHERE r.score_value_for_money IS NOT NULL
    """
    params = {}
    if city:
        query_base += """
        MATCH (h)-[:LOCATED_IN]->(c:City) 
        WHERE toLower(c.name) = toLower($city) 
        """
        params["city"] = city
    elif country:
        query_base += """
        MATCH (h)-[:LOCATED_IN]->(:City)-->(cntry:Country) 
        WHERE toLower(cntry.name) = toLower($country) 
        """
        params["country"] = country
    query_end = """
    WITH h.name AS Hotel, AVG(r.score_value_for_money) AS AvgScore 
    RETURN Hotel, AvgScore 
    ORDER BY AvgScore DESC 
    LIMIT 5
    """
    with driver.session() as session:
        result = session.run(query_base + query_end, **params)
        data = [f"Hotel: {record['Hotel']} (Value Score: {round(record['AvgScore'], 2)})" for record in result]
        return (data if data else ["No value data found."]), query_base + query_end

def get_best_hotels_by_location(driver, city=None, country=None):
    query_base = """
    MATCH (r:Review)-[:REVIEWED]->(h:Hotel) 
    WHERE r.score_location IS NOT NULL
    """
    params = {}
    if city:
        query_base += """
        MATCH (h)-[:LOCATED_IN]->(c:City) 
        WHERE toLower(c.name) = toLower($city) 
        """
        params["city"] = city
    elif country:
        query_base += """
        MATCH (h)-[:LOCATED_IN]->(:City)-->(cntry:Country) 
        WHERE toLower(cntry.name) = toLower($country) 
        """
        params["country"] = country
    query_end = """
    WITH h.name AS Hotel, AVG(r.score_location) AS AvgScore 
    RETURN Hotel, AvgScore 
    ORDER BY AvgScore DESC 
    LIMIT 5
    """
    with driver.session() as session:
        result = session.run(query_base + query_end, **params)
        data = [f"Hotel: {record['Hotel']} (Location Score: {round(record['AvgScore'], 2)})" for record in result]
        return (data if data else ["No location data found."]), query_base + query_end

def get_best_hotels_by_comfort(driver, city=None, country=None):
    query_base = """
    MATCH (r:Review)-[:REVIEWED]->(h:Hotel) 
    WHERE r.score_comfort IS NOT NULL
    """
    params = {}
    if city:
        query_base += """
        MATCH (h)-[:LOCATED_IN]->(c:City) 
        WHERE toLower(c.name) = toLower($city) 
        """
        params["city"] = city
    elif country:
        query_base += """
        MATCH (h)-[:LOCATED_IN]->(:City)-->(cntry:Country) 
        WHERE toLower(cntry.name) = toLower($country) 
        """
        params["country"] = country
    query_end = """
    WITH h.name AS Hotel, AVG(r.score_comfort) AS AvgScore 
    RETURN Hotel, AvgScore 
    ORDER BY AvgScore DESC 
    LIMIT 5
    """
    with driver.session() as session:
        result = session.run(query_base + query_end, **params)
        data = [f"Hotel: {record['Hotel']} (Comfort Score: {round(record['AvgScore'], 2)})" for record in result]
        return (data if data else ["No comfort data found."]), query_base + query_end


def generate_response(user_query, detected_entities, text_retriever, feature_retriever, driver, llm_client, retrieval_mode, llm_model_name):
    context_parts = []
    executed_queries = []
    visualization_data = [] 
    
    if retrieval_mode in ["Baseline (Graph Only)", "Hybrid (Graph + Embeddings)"]:
        city = detected_entities.get("City")
        if city:
            graph_results, query_str = get_hotels_by_city(driver, city)
            executed_queries.append(f"### Intent: City Search ({city})\n{query_str}")
            if graph_results: 
                context_parts.append(f"Top Hotels in {city} (from KG):\n" + "\n".join(graph_results))
                visualization_data.extend(get_visualization_data(driver, "City", detected_entities))

        country = detected_entities.get("Country")
        if country:
            graph_results, query_str = get_hotels_by_country(driver, country)
            executed_queries.append(f"### Intent: Country Search ({country})\n{query_str}")
            if graph_results: 
                context_parts.append(f"Top Hotels in {country} (from KG):\n" + "\n".join(graph_results))
                visualization_data.extend(get_visualization_data(driver, "Country", detected_entities))

        t_type = detected_entities.get("Traveller Type")
        if t_type:
            if "country" in user_query.lower() or "countries" in user_query.lower():
                country_results, query_str = get_best_countries_by_traveller_type(driver, t_type)
                executed_queries.append(f"### Intent: Country by Traveller Type ({t_type})\n{query_str}")
                if country_results: 
                    context_parts.append(f"Top Rated Countries for '{t_type}':\n" + "\n".join(country_results))
            else:
                type_results, query_str = get_hotels_by_traveller_type(driver, t_type)
                executed_queries.append(f"### Intent: Hotel by Traveller Type ({t_type})\n{query_str}")
                if type_results: 
                    context_parts.append(f"Top Hotels for '{t_type}':\n" + "\n".join(type_results))
                    visualization_data.extend(get_visualization_data(driver, "Traveller Type", detected_entities))

        gender = detected_entities.get("Gender")
        if gender:
            gender_results, query_str = get_top_countries_by_gender(driver, gender)
            executed_queries.append(f"### Intent: Gender Demographics ({gender})\n{query_str}")
            if gender_results: 
                context_parts.append(f"Popular for {gender}:\n" + "\n".join(gender_results))
                visualization_data.extend(get_visualization_data(driver, "Gender", detected_entities))

        age_groups = detected_entities.get("Target Age Groups")
        if age_groups:
            if "country" in user_query.lower() or "countries" in user_query.lower():
                age_results, query_str = get_top_countries_by_age_groups(driver, age_groups)
                executed_queries.append(f"### Intent: Country by Age Group ({age_groups})\n{query_str}")
                if age_results: context_parts.append(f"Popular Countries for Ages {age_groups}:\n" + "\n".join(age_results))
            else:
                hotel_results, query_str = get_top_hotels_by_age_groups(driver, age_groups)
                executed_queries.append(f"### Intent: Hotel by Age Group ({age_groups})\n{query_str}")
                if hotel_results: 
                    context_parts.append(f"Top Rated Hotels for Ages {age_groups}:\n" + "\n".join(hotel_results))
                    visualization_data.extend(get_visualization_data(driver, "Age Group", detected_entities))

        sort_type = detected_entities.get("Sort By")
        target_city = detected_entities.get("City")
        target_country = detected_entities.get("Country")
        location_str = target_city if target_city else (target_country if target_country else "Global")

        if sort_type:
            def handle_sort(func, label):
                res, q = func(driver, city=target_city, country=target_country)
                executed_queries.append(f"### Intent: Sort by {label} ({location_str})\n{q}")
                context_parts.append(f"{label} Hotels ({location_str}):\n" + "\n".join(res))
            
            if sort_type == "Cleanliness": handle_sort(get_best_hotels_by_cleanliness, "Cleanest")
            elif sort_type == "Value": handle_sort(get_best_hotels_by_value, "Best Value")
            elif sort_type == "Location": handle_sort(get_best_hotels_by_location, "Best Located")
            elif sort_type == "Comfort": handle_sort(get_best_hotels_by_comfort, "Most Comfortable")
            
            if not visualization_data:
                if target_city:
                     visualization_data.extend(get_visualization_data(driver, "City", detected_entities))
                elif target_country:
                     visualization_data.extend(get_visualization_data(driver, "Country", detected_entities))
                else:
                     visualization_data.extend(get_visualization_data(driver, "Sort By", detected_entities))

    if retrieval_mode in ["Embeddings (Vector Only)", "Hybrid (Graph + Embeddings)"]:
        vector_results = text_retriever.invoke(user_query)
        text_reviews = "\n".join([doc.page_content for doc in vector_results])
        context_parts.append(f"Relevant Text Reviews:\n{text_reviews}")

        if feature_retriever:
            feature_results = feature_retriever.invoke(user_query)
            
            unique_feature_docs = []
            seen_hotel_ids = set()
            
            for doc in feature_results:
                match = re.search(r"\(ID:\s*([^)]+)\)", doc.page_content)
                if match:
                    hotel_id = match.group(1)
                    if hotel_id not in seen_hotel_ids:
                        seen_hotel_ids.add(hotel_id)
                        unique_feature_docs.append(doc)
                else:
                    unique_feature_docs.append(doc)
            
            feature_text = "\n".join([doc.page_content for doc in unique_feature_docs])
            context_parts.append(f"Relevant Hotel Feature Profiles (Scores from Neo4j):\n{feature_text}")
    
    full_context = "\n\n".join(context_parts)
    if not full_context: full_context = "No relevant data found in the selected retrieval source."

    entity_context_str = ", ".join([f"{k}: {v}" for k,v in detected_entities.items()])
    
    template = f"""
    You are a Hotel Recommender.
    RETRIEVAL MODE: {retrieval_mode}
    DETECTED ENTITIES: {entity_context_str if entity_context_str else "None"}
    CONTEXT:
    {{context}}
    USER QUESTION: {{question}}
    Answer:
    """
    prompt = PromptTemplate.from_template(template).format(context=full_context, question=user_query)
    
    llm = HuggingFaceLLMWrapper(client=llm_client, model_name=llm_model_name)
    return llm.invoke(prompt), full_context, executed_queries, visualization_data


#Streemlit
st.title("🏨 Graph-RAG Hotel Assistant")
st.sidebar.header("Configuration")
retrieval_mode = st.sidebar.radio("Select Retrieval Strategy:", ("Hybrid (Graph + Embeddings)", "Baseline (Graph Only)", "Embeddings (Vector Only)"))
st.sidebar.markdown("---")
embedding_model_choice = st.sidebar.selectbox("Select Embedding Model:", list(EMBEDDING_MODELS.keys()))
selected_embedding_repo = EMBEDDING_MODELS[embedding_model_choice]
st.sidebar.markdown("---")
llm_model_choice = st.sidebar.selectbox("Select LLM Model:", list(LLM_MODELS.keys()))
selected_llm_repo = LLM_MODELS[llm_model_choice]

user_query = st.text_input("Ask me:", placeholder="e.g., Best hotels for people aged 18-24?")

if st.button("Ask Assistant"):
    try:
        with st.spinner(f"Processing with {llm_model_choice} using {retrieval_mode}..."):
            text_retriever = setup_text_vector_store(selected_embedding_repo)
            feature_retriever = setup_feature_vector_store_neo4j(embedding_model_choice, selected_embedding_repo)
            driver = setup_graph_db()
            entity_db = get_all_entities(driver)
            llm_client = InferenceClient(model=selected_llm_repo, token=HF_TOKEN)
        
        detected_entities = extract_entities_from_query(user_query, entity_db)
        if detected_entities: st.info(f"Entities: {detected_entities}")
        
        answer, context, queries, viz_data = generate_response(
            user_query, detected_entities, text_retriever, feature_retriever, driver, llm_client, retrieval_mode, llm_model_choice
        )
        
        st.success("Recommendation:")
        st.write(answer)
        
        if queries:
            with st.expander("Cypher Queries Executed"):
                for q in queries: st.code(q, language='cypher')

        if viz_data:
            with st.expander("Graph Visualization Snippets"):
                graph = graphviz.Digraph()
                added_edges = set()
                for source, rel, target in viz_data:
                    edge_key = (source, target)
                    if edge_key not in added_edges:
                        graph.edge(source, target, label=rel)
                        added_edges.add(edge_key)
                st.graphviz_chart(graph)
        
        with st.expander("Debug Context"):
            st.text(context)
            
    except Exception as e:
        st.error(f"Error: {e}")