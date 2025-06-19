import faiss
import numpy as np
from rake_nltk import Rake

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index

def extract_keywords(lecture_plan):
    rake = Rake()
    rake.extract_keywords_from_text(lecture_plan)
    keywords = rake.get_ranked_phrases()
    return keywords


def query_faiss_index(index, topic_embeddings, chunks, top_k=3):
    query_vectors = topic_embeddings.astype('float32')
    
    if query_vectors.ndim == 1:
        query_vectors = query_vectors.reshape(1, -1)
    
    assert query_vectors.shape[1] == index.d, "Dimension mismatch between query vectors and index."
    _, indices = index.search(query_vectors, top_k)
    relevant_chunks = []
    for idx_list in indices:
        for idx in idx_list:
            relevant_chunks.append(chunks[idx])
    return relevant_chunks