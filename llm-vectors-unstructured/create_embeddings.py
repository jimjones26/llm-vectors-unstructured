import os
from dotenv import load_dotenv
from google import genai

load_dotenv(override=True)


client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

result = client.models.embed_content(
    model="gemini-embedding-exp-03-07", contents="What is the meaning of life?"
)

print(result.embeddings)
