import json
import os

import numpy as np
import redis
from dotenv import find_dotenv, load_dotenv
from langchain.docstore.document import Document
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer

load_dotenv(find_dotenv())


INDEX_NAME = "balic"


def _prepare_query(k: int = 4) -> Query:
    # Prepare the Query
    # '*' means we perform similarity search over an entire vector field.
    # Other option is to run similarity query on the result of the
    # primary filter query - <primary_filter_query>=>[<vector_similarity_query>]
    hybrid_fields = "*"
    # k: top k results
    # @content_vector: name of the vector field
    # $vector: the query vector as blob
    # AS vector_score: name of the distance field
    base_query = f"{hybrid_fields}=>[KNN {k} @content_vector $vector AS vector_score]"
    return_fields = ["metadata", "content", "vector_score"]
    return (
        Query(base_query)
        .return_fields(*return_fields)
        .sort_by("vector_score")
        .paging(0, k)
        # Dialect refers to how redis queries are parsed.
        # It is necessary to set dialect to 2 or greater
        # for vector similarity search.
        # Sources:
        # https://redis.io/docs/stack/search/reference/query_syntax/
        .dialect(2)
    )


client = SentenceTransformer("all-mpnet-base-v2")

query = input("Enter query: ")
query = query.replace("\n", " ")
query_embedding = client.encode(query).tolist()

redis_query = _prepare_query()

params_dict = {"vector": np.array(query_embedding).astype(dtype=np.float32).tobytes()}

redis_client = redis.from_url(os.getenv("REDIS_URL"))

# Perform vector search
results = redis_client.ft(INDEX_NAME).search(redis_query, params_dict)

# Prepare document results
docs = [
    (
        Document(page_content=result.content, metadata=json.loads(result.metadata)),
        float(result.vector_score),
    )
    for result in results.docs
]

for doc in docs:
    print(doc)
