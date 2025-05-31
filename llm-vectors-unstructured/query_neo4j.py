import os
from dotenv import load_dotenv
from google import genai
from langchain_neo4j import Neo4jGraph

load_dotenv(override=True)

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

response = client.models.embed_content(
    model="models/embedding-001", contents="What does Hallucination mean?"
)

embedding = (
    response.embeddings[0].values
    if response.embeddings and len(response.embeddings) > 0
    else None
)

print(embedding)

# Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

# Run query
result = graph.query(
    """
CALL db.index.vector.queryNodes('chunkVector', 6, $embedding)
YIELD node, score
RETURN node.text, score
""",
    {"embedding": embedding},
)

# Display results
for row in result:
    print(row["node.text"], row["score"])
