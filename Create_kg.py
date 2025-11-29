import pandas as pd
from neo4j import GraphDatabase
import os
import time

def load_config(config_file='config.txt'):
    config = {}
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file '{config_file}' not found.")
    
    with open(config_file, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                config[key] = value
    return config

#Knowledge Graph 
class HotelGraphBuilder:
    def __init__(self):
        config = load_config()
        self.driver = GraphDatabase.driver(
            config.get('URI', 'neo4j://localhost:7687'), 
            auth=(config.get('USERNAME', 'neo4j'), config.get('PASSWORD', 'password'))
        )

    def close(self):
        self.driver.close()

    def clear_database(self):
        print("Cleaning existing database...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("Database cleared.")

    def create_constraints(self):
        print("Creating constraints...")
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Traveller) REQUIRE t.user_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (h:Hotel) REQUIRE h.hotel_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:City) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (k:Country) REQUIRE k.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Review) REQUIRE r.review_id IS UNIQUE"
        ]
        with self.driver.session() as session:
            for q in queries:
                session.run(q)


    def ingest_hotels_data(self):
        print("Processing Hotels, Cities, and Countries...")
        hotels_df = pd.read_csv('Dataset/hotels.csv')
        reviews_df = pd.read_csv('Dataset/reviews.csv')

        avg_scores = reviews_df.groupby('hotel_id')['score_overall'].mean().reset_index()
        avg_scores.columns = ['hotel_id', 'average_reviews_score']
        
        hotels_df = pd.merge(hotels_df, avg_scores, on='hotel_id', how='left')
        hotels_df['average_reviews_score'] = hotels_df['average_reviews_score'].fillna(0)

        hotels_df.rename(columns={'hotel_name': 'name'}, inplace=True)

        data = hotels_df.to_dict('records')

        query = """
        UNWIND $rows AS row
        MERGE (co:Country {name: row.country})
        MERGE (ci:City {name: row.city})
        MERGE (ci)-[:LOCATED_IN]->(co)
        
        MERGE (h:Hotel {hotel_id: row.hotel_id})
        SET h.name = row.name,
            h.star_rating = row.star_rating,
            h.cleanliness_base = row.cleanliness_base,
            h.comfort_base = row.comfort_base,
            h.facilities_base = row.facilities_base,
            h.average_reviews_score = row.average_reviews_score
            
        MERGE (h)-[:LOCATED_IN]->(ci)
        """
        
        self._run_batch(query, data)

    def ingest_travellers(self):
        print("Processing Travellers...")
        users_df = pd.read_csv('Dataset/users.csv')

        users_df.rename(columns={
            'traveller_type': 'type',
            'user_gender': 'gender',
            'age_group': 'age'
        }, inplace=True)

        data = users_df.to_dict('records')

        query = """
        UNWIND $rows AS row
        MERGE (t:Traveller {user_id: row.user_id})
        SET t.age = row.age,
            t.type = row.type,
            t.gender = row.gender

        MERGE (c:Country {name: row.country})
        MERGE (t)-[:FROM_COUNTRY]->(c)
        """
        
        self._run_batch(query, data)

    def ingest_reviews(self):
        print("Processing Reviews and History...")
        reviews_df = pd.read_csv('Dataset/reviews.csv')

        reviews_df.rename(columns={
            'review_date': 'date',
            'review_text': 'text'
        }, inplace=True)

        batch_size = 5000
        data = reviews_df.to_dict('records')
        total = len(data)

        query = """
        UNWIND $rows AS row
        MATCH (t:Traveller {user_id: row.user_id})
        MATCH (h:Hotel {hotel_id: row.hotel_id})

        MERGE (r:Review {review_id: row.review_id})
        SET r.text = row.text,
            r.date = row.date,
            r.score_overall = row.score_overall,
            r.score_cleanliness = row.score_cleanliness,
            r.score_comfort = row.score_comfort,
            r.score_facilities = row.score_facilities,
            r.score_location = row.score_location,
            r.score_staff = row.score_staff,
            r.score_value_for_money = row.score_value_for_money
        
        MERGE (t)-[:WROTE]->(r)
        MERGE (r)-[:REVIEWED]->(h)
        
        MERGE (t)-[:STAYED_AT]->(h)
        """

        for i in range(0, total, batch_size):
            batch = data[i:i+batch_size]
            self._run_batch(query, batch)
            print(f"   Processed reviews {i} to {min(i+batch_size, total)}")

    def ingest_visa_requirements(self):
        print("Processing Visa Requirements...")
        visa_df = pd.read_csv('Dataset/visa.csv')

        visa_required = visa_df[visa_df['requires_visa'].isin(['Yes', '1', 'True'])].copy()

        data = visa_required.to_dict('records')

        query = """
        UNWIND $rows AS row
        MERGE (c1:Country {name: row.from})
        MERGE (c2:Country {name: row.to})
        MERGE (c1)-[v:NEEDS_VISA]->(c2)
        SET v.visa_type = row.visa_type
        """
        
        self._run_batch(query, data)

    def _run_batch(self, query, data):
        with self.driver.session() as session:
            session.run(query, rows=data)

#Main
if __name__ == "__main__":
    builder = None
    try:
        required_files = ['Dataset/hotels.csv', 'Dataset/reviews.csv', 'Dataset/users.csv', 'Dataset/visa.csv', 'config.txt']
        missing = [f for f in required_files if not os.path.exists(f)]
        
        if missing:
            print(f"Error: Missing files: {missing}")
        else:
            start_time = time.time()
            builder = HotelGraphBuilder()
            
            builder.clear_database()
            builder.create_constraints()
            builder.ingest_hotels_data()
            builder.ingest_travellers()
            builder.ingest_reviews()
            builder.ingest_visa_requirements()
            
            print(f"\nKnowledge Graph successfully built in {time.time() - start_time:.2f} seconds!")
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if builder:
            builder.close()