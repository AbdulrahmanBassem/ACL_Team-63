# 🏨 Graph-RAG Travel Assistant & Hotel Recommender

An end-to-end **Graph Retrieval-Augmented Generation (Graph-RAG)** system designed to provide personalized travel and hotel recommendations. By combining the structured reasoning of a **Neo4j Knowledge Graph** with the semantic understanding of **Vector Embeddings**, this system delivers grounded, factual, and explainable answers to complex user queries.

## 🚀 Key Features

* **Hybrid Retrieval (Graph-RAG):** Combines symbolic graph traversal (Cypher queries) with semantic vector search (Embeddings) for high-precision retrieval.
* **Multi-Model Support:**
    * **LLMs:** Switch dynamically between **Gemma 2 (2B)**, **Mistral (v0.2)**, and **Llama 3 (8B)**.
    * **Embeddings:** Choose between **MiniLM (Speed)** and **MPNet (Accuracy)**.
* **Dynamic Graph Visualization:** Real-time generation of knowledge graph snippets (using Graphviz) to visualize the reasoning behind recommendations.
* **Intent Understanding:** Specialized logic to detect intents like *City Search*, *Country Search*, *Traveller Type*, *Demographics*, and *Sorting* (Cleanliness, Value, Location, Comfort).
* **Transparency:** View the exact Cypher queries executed and the raw context retrieved from the database.

---

## 🛠️ System Architecture

The pipeline consists of three main stages:

1.  **Knowledge Graph Construction (`Create_kg.py`):** Ingests structured data (Hotels, Users, Reviews, Visa rules) into a Neo4j Graph Database.
2.  **Vector Indexing (`create_embeddings.py`):** Generates semantic embeddings for hotel reviews, deduplicates them, and stores them directly in Neo4j as a Vector Index.
3.  **Inference Application (`app.py`):** A Streamlit interface that processes user queries, performs hybrid retrieval, and prompts the LLM via the Hugging Face Inference API.

---

## 📋 Prerequisites

Before running the project, ensure you have the following installed:

1.  **Python 3.8+**
2.  **Neo4j Desktop or AuraDB:** A running Neo4j instance.
3.  **Graphviz:** Required for graph visualization.
    * *Windows:* [Download Installer](https://graphviz.org/download/) (Important: Add Graphviz to system PATH during installation).
    * *Mac:* `brew install graphviz`
    * *Linux:* `sudo apt-get install graphviz`
4.  **Hugging Face Account:** You need an Access Token to use the Inference API.

---

## ⚙️ Configuration

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-repo/graph-rag-hotel-recommender.git](https://github.com/your-repo/graph-rag-hotel-recommender.git)
    cd graph-rag-hotel-recommender
    ```

2.  **Install Python Dependencies:**
    ```bash
    pip install streamlit pandas neo4j langchain langchain-community langchain-huggingface sentence-transformers graphviz
    ```

3.  **Setup `config.txt`:**
    Create a file named `config.txt` in the root directory. It **must** follow this exact format (key=value, no spaces around `=`):

    ```text
    URI=neo4j://127.0.0.1:7687
    USERNAME=neo4j
    PASSWORD=your_neo4j_password
    HFToken=hf_your_hugging_face_token_here
    ```

---

## 🏃‍♂️ How to Run the Pipeline

Follow these steps in order to set up the data and run the app.

### Step 1: Build the Knowledge Graph
Populate your Neo4j database with the base nodes (Hotels, Cities, Travellers) and relationships.
*Warning: This script clears existing data in the database before writing.*

```bash
python Create_kg.py

##Step 2: Generate Vector Embeddings

Compute embeddings for hotel reviews and store them in Neo4j vector indices.  
This script batches the data to ensure stability.

```bash
python create_embeddings.py
```

**Note:**  
This creates two separate indices in Neo4j:
- One for **MiniLM**
- One for **MPNet**

---

## Step 3: Launch the Application

Start the Streamlit web interface.

```bash
streamlit run app.py
```

---

## 🖥️ Usage Guide

### Configuration Sidebar

- **Retrieval Strategy**  
  Choose **Hybrid (Graph + Embeddings)** for the best results.  
  You can also test:
  - Baseline (Graph Only)
  - Vector Only

- **Embedding Model**  
  Select which pre-computed index to use:
  - MiniLM
  - MPNet

- **LLM Model**  
  Select the generator model (e.g., *Mistral 7B*).

---

### Ask Questions

Examples:

- **Simple**  
  `Find hotels in London.`

- **Complex**  
  `Best hotels in Egypt for solo travellers that are clean and comfortable.`

- **Demographic**  
  `Where do people aged 25–34 usually visit?`

- **Sorting**  
  `Cleanest hotels in Paris.`

---

### Analyze Results

- **Recommendation**  
  Read the LLM's generated answer.

- **Cypher Queries Executed**  
  Expand this section to see the exact database queries used.

- **Graph Visualization Snippets**  
  Expand this to see the visual subgraph of the results (nodes and edges).

- **Debug Context**  
  View the raw text passed to the LLM.

---

## 📂 Project Structure

```plaintext
├── Dataset/                 # CSV and SQLite data source files
│   ├── hotels.csv
│   ├── reviews.csv
│   ├── users.csv
│   └── visa.csv
├── app.py                   # Main Streamlit application
├── Create_kg.py             # Script to populate Neo4j Knowledge Graph
├── create_embeddings.py     # Script to generate and store embeddings
├── config.txt               # Configuration file (ignored by git)
├── requirements.txt         # Project dependencies
└── README.md                # Documentation
```

---

## ⚠️ Troubleshooting

- **OSError: No data / Connection Failed**  
  This usually happens during embedding generation if the batch size is too large.  
  The current script uses `BATCH_SIZE = 200`.  
  If this persists, verify your Neo4j connection string in `config.txt`.

- **Graph Visualization Not Showing**  
  Ensure **Graphviz** is installed on your OS and added to the system `PATH`.  
  The Python library alone is not enough.

- **410 Client Error (LLM)**  
  If an LLM model fails (e.g., *Model is deprecated*), switch to a different model in the sidebar  
  (e.g., **Mistral v0.2** or **Zephyr**).

- **Missing Graph in UI**  
  The graph visualization requires specific entities  
  (City, Country, Traveller Type, etc.) to be detected.  
  Generic questions may not generate a visual graph.
