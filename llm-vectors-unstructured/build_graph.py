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
        """,
        data,
    )


# Create Genai object
llm = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Connect to Neo4j
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
)
driver.verify_connectivity()

for chunk in chunks:
    with driver.session(database="neo4j") as session:
        session.execute_write(create_chunk, get_course_data(llm, chunk))

driver.close()
