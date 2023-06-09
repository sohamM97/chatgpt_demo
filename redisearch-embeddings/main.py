import json
import os
import uuid
from collections import deque

import numpy as np
import redis
from dotenv import load_dotenv
from langchain.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)
from langchain.text_splitter import NLTKTextSplitter
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from sentence_transformers import SentenceTransformer

load_dotenv("../.env")

INDEX_NAME = "balic"


def load_documents(document_path: str):
    source_folder = deque()
    source_folder.append(document_path)
    documents = []

    while len(source_folder) != 0:
        folder = source_folder.popleft()
        for i in os.listdir(folder):
            file_path = os.path.join(folder, i)
            if os.path.isdir(file_path):
                source_folder.append(file_path)
            elif os.path.isfile(file_path):
                if i.endswith(".xlsx") or i.endswith(".xls"):
                    documents.append(UnstructuredExcelLoader(file_path).load())
                elif i.endswith(".csv"):
                    documents.append(CSVLoader(file_path).load())
                elif i.endswith(".pptx"):
                    documents.append(UnstructuredPowerPointLoader(file_path).load())
                elif i.endswith(".docx") or i.endswith(".doc"):
                    documents.append(Docx2txtLoader(file_path).load())
                elif i.endswith(".pdf"):
                    documents.append(PyPDFLoader(file_path).load())

    return documents


def _check_index_exists(client, index_name: str) -> bool:
    """Check if Redis index exists."""
    try:
        client.ft(index_name).info()
    except Exception:
        print("Index does not exist")
        return False
    print("Index already exists")
    return True


def _redis_key(prefix: str) -> str:
    """Redis key schema for a given prefix."""
    return f"{prefix}:{uuid.uuid4().hex}"


document_path = "/home/soham/Downloads/balic_docs"
documents = load_documents(document_path)

text_splitter = NLTKTextSplitter(chunk_size=300, chunk_overlap=50)
texts = []
sources = []
for document in documents:
    for para in document:
        paras = text_splitter.split_text(para.page_content)
        texts.extend(paras)
        sources.extend(
            [
                {
                    **para.metadata,
                    "source": para.metadata["source"][len(document_path) + 1 :],
                }
                for _ in range(len(paras))
            ]
        )

# source: https://www.sbert.net/docs/pretrained_models.html
client = SentenceTransformer("all-mpnet-base-v2")

texts = [text.replace("\n", " ") for text in texts]
doc_embeddings = client.encode(texts).tolist()

dim = len(doc_embeddings[0])

redis_client = redis.from_url(os.getenv("REDIS_URL"))

# Sources:
# https://youtu.be/infTV4ifNZY
# https://redis.io/docs/stack/search/reference/vectors/

prefix = f"doc:{INDEX_NAME}"
if not _check_index_exists(redis_client, index_name=INDEX_NAME):
    # Define schema - fields to be indexed are content, metadata and content_vector
    schema = (
        TextField(name="content"),
        TextField(name="metadata"),
        VectorField(
            name="content_vector",
            # FLAT is a brute force algorithm to perform KNN similarity search
            # i.e. compute distances of query point to all other data
            # points in the dataset
            algorithm="FLAT",
            attributes={
                "TYPE": "FLOAT32",
                # DIM is the dimension of the vector,
                # 768 in case of all-mpnet-base-v2 sentence transformer
                "DIM": dim,
                "DISTANCE_METRIC": "COSINE",
            },
        ),
    )

    # Create Redis Index
    # FT means full-text and is a prefix used for RediSearch commands
    redis_client.ft(INDEX_NAME).create_index(
        # creates an index on keys prefixed docs:balic:,
        # on fields content, metadata, and content_vector
        fields=schema,
        definition=IndexDefinition(prefix=[prefix], index_type=IndexType.HASH),
    )

ids = []

# Write data to redis
batch_size = 1000
pipeline = redis_client.pipeline(transaction=False)
for i, text in enumerate(texts):
    key = _redis_key(prefix)
    metadata = sources[i]
    embedding = doc_embeddings[i]
    pipeline.hset(
        key,
        mapping={
            "content": text,
            "content_vector": np.array(embedding, dtype=np.float32).tobytes(),
            "metadata": json.dumps(metadata),
        },
    )
    ids.append(key)

    # Write batch
    if i % batch_size == 0:
        pipeline.execute()

# Cleanup final batch
pipeline.execute()
print(ids)
