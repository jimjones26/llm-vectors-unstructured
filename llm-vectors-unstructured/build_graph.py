import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from google import genai
from neo4j import GraphDatabase
from textblob import TextBlob

load_dotenv(override=True)

COURSES_PATH = "./data/asciidoc"

loader = DirectoryLoader(COURSES_PATH, glob="**/lesson.adoc", loader_cls=TextLoader)
print("Loading documents...")
docs = loader.load()
print(f"Loaded {len(docs)} documents.")

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1500,
    chunk_overlap=200,
)

print("Splitting documents into chunks...")
chunks = text_splitter.split_documents(docs)
print("Number of chunks:", len(chunks))


# Create a function to get the embedding
def get_embedding(llm, text):
    response = llm.models.embed_content(model="models/embedding-001", contents=text)
    embedding = (
        response.embeddings[0].values
        if response.embeddings and len(response.embeddings) > 0
        else None
    )
    return embedding


# Create a function to get the course data
def get_course_data(llm, chunk):
    data = {}

    path = chunk.metadata["source"].split(os.path.sep)

    data["course"] = path[-6]
    data["module"] = path[-4]
    data["lesson"] = path[-2]
    data["url"] = (
        f"https://graphacademy.neo4j.com/courses/{data['course']}/{data['module']}/{data['lesson']}"
    )
    data["text"] = chunk.page_content
    data["embedding"] = get_embedding(llm, data["text"])
    data["topics"] = TextBlob(data["text"]).noun_phrases

    return data


def create_chunk(tx, data):
    tx.run(
        """
        MERGE (c:Course {name: $course})
        MERGE (c)-[:HAS_MODULE]->(m:Module{name: $module})
        MERGE (m)-[:HAS_LESSON]->(l:Lesson{name: $lesson, url: $url})
        MERGE (l)-[:CONTAINS]->(p:Paragraph{text: $text})
        WITH p
        CALL db.create.setNodeVectorProperty(p, "embedding", $embedding)
           
        FOREACH (topic in $topics |
            MERGE (t:Topic {name: topic})
            MERGE (p)-[:MENTIONS]->(t)
        )
        """,
        data,
    )


# Create Genai object
print("Creating GenAI client...")
llm = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Connect to Neo4j
print("Connecting to Neo4j...")
neo4j_uri = os.getenv("NEO4J_URI")
if not neo4j_uri:
    raise ValueError("NEO4J_URI environment variable is not set.")
driver = GraphDatabase.driver(
    neo4j_uri,
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
)
driver.verify_connectivity()
print("Connected to Neo4j.")

print("Processing chunks and writing to Neo4j...")
for idx, chunk in enumerate(chunks, 1):
    print(f"Processing chunk {idx}/{len(chunks)}...")
    with driver.session(database="neo4j") as session:
        session.execute_write(create_chunk, get_course_data(llm, chunk))

print("All chunks processed. Closing Neo4j connection.")
driver.close()
print("Done.")
