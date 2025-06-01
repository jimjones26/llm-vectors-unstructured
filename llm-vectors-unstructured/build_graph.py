import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from google import genai
from neo4j import GraphDatabase

load_dotenv(override=True)

COURSES_PATH = "./data/asciidoc"

loader = DirectoryLoader(COURSES_PATH, glob="**/lesson.adoc", loader_cls=TextLoader)
docs = loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1500,
    chunk_overlap=200,
)

chunks = text_splitter.split_documents(docs)
print("number of chunks: ", len(chunks))


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


# Create OpenAI object

# Connect to Neo4j

# Create a function to run the Cypher query

# Iterate through the chunks and create the graph

# Close the neo4j driver
