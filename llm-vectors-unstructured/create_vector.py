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
# print(chunks)

# Create a Neo4j vector store
neo4j_db = Neo4jVector.from_documents(
    chunks,
    GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=SecretStr(GOOGLE_API_KEY) if GOOGLE_API_KEY else None,
    ),
    url=os.environ.get("NEO4J_URI"),
    username=os.environ.get("NEO4J_USERNAME"),
    password=os.environ.get("NEO4J_PASSWORD"),
    database=os.environ.get("NEO4J_DATABASE"),
    index_name="chunkVector",
    node_label="Chunk",
    text_node_property="text",
    embedding_node_property="embedding",
)
