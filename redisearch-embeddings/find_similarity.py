import json
import os

import numpy as np
import redis
from langchain.docstore.document import Document
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer

INDEX_NAME = "balic"


def _prepare_query(k: int = 4) -> Query:
    # Prepare the Query
    hybrid_fields = "*"
    base_query = f"{hybrid_fields}=>[KNN {k} @content_vector $vector AS vector_score]"
    return_fields = ["metadata", "content", "vector_score"]
    return (
        Query(base_query)
        .return_fields(*return_fields)
        .sort_by("vector_score")
        .paging(0, k)
        .dialect(2)
    )


client = SentenceTransformer("all-mpnet-base-v2")

query = input("Enter query: ")
query = query.replace("\n", " ")
query_embedding = client.encode(query).tolist()

redis_query = _prepare_query()

print(len(query_embedding))
print(redis_query.query_string())

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
