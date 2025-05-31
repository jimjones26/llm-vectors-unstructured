import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_neo4j import Neo4jVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr

load_dotenv(override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

COURSES_PATH = "./data/asciidoc"

# Load lesson documents
loader = DirectoryLoader(COURSES_PATH, glob="**/lesson.adoc", loader_cls=TextLoader)
docs = loader.load()

# Create a text splitter
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1500,
    chunk_overlap=200,
)

# Split documents into chunks
chunks = text_splitter.split_documents(docs)
print(f"Split {len(docs)} documents into {len(chunks)} chunks.")

# 1. Initialize the vector store with the first chunk
print("Initializing Neo4jVector store...")
neo4j_db = Neo4jVector.from_documents(
    chunks[:1],
    GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=SecretStr(GOOGLE_API_KEY) if GOOGLE_API_KEY else None,
    ),
    url=os.environ.get("NEO4J_URI"),
    username=os.environ.get("NEO4J_USERNAME"),
    password=os.environ.get("NEO4J_PASSWORD"),
    database="neo4j",
    index_name="chunkVector",
    node_label="Chunk",
    text_node_property="text",
    embedding_node_property="embedding",
)

# 2. Add the rest of the chunks in smaller batches
batch_size = 20  # Adjust batch size as needed based on API limits
total_chunks = len(chunks)

for i in range(1, total_chunks, batch_size):
    batch = chunks[i : i + batch_size]
    print(
        f"Processing batch {i // batch_size + 1}: chunks {i + 1}-{min(i + batch_size, total_chunks)} of {total_chunks}"
    )
    neo4j_db.add_documents(batch)

print("Vector store creation complete.")
