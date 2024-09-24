from helper_utils import word_wrap
from pypdf import PdfReader
import os
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

reader = PdfReader("data")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

pdf_texts = [texts for texts in pdf_texts if text]

# split the text into smaller chunks
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

character_splitter = RecursiveCharacterTextSplitter(
    seperators=["\n\n","\n","."," ",""], chunk_size=1000, chunk_overlap=0
)

character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap = 0, token_size = 256
)

token_split_texts = []

for text in character_split_texts:
    token_split_texts+=token_splitter.split_text(text)

# start embedding process

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedding_function = SentenceTransformerEmbeddingFunction()

chroma_client = chromadb.Client()
chroma_collection = chrom_client.create_collection(
    "microsoft-collection", embedding_function = embedding_function
)

# extract embeddings of the token_split_text
ids = [str(i) for i in range(len(token_split_texts))]

chroma_collection.add(ids=ids, documents=token_split_texts)
chroma_collection.count()

query="What was total revenue for the year?"

results = chroma_collection.query(query_texts=[query], n_results=5)
retrieved_documents = results["documents"][0]

def augment_query_generated(query, model="gpt-3.5-turbo"):
    prompt = """You are a helpful expert financial research assistant. 
   Provide an example answer to the given question, that might be found in a document like an annual report."""
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content

original_query = "What was the total profit for the year, and how does it compare to the previous year?"
hypothetical_answer = augment_query_generated(original_query)

joint_query = f"{original_query} {hypothetical_answer}"
print(word_wrap(joint_query))

results = chroma_collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "embeddings"]
)
retrieved_documents = results["documents"][0]

embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = umap_transform.transform(embeddings)

