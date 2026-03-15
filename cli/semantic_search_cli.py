#!/usr/bin/env python3

import argparse
import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np

datafile = 'data/movies.json'
# datafile = 'data/killshot.json'
# datafile = 'data/killshot2.json'
# datafile = 'data/movie1.json'
# datafile = 'data/two_movies.json'

class SemanticSearch:
    def __init__(self) -> None:
        self.name = "all-MiniLM-L6-v2"
        self.model = SentenceTransformer(self.name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if len(text.strip()) == 0:
            raise ValueError("Empty text")
        e = self.model.encode([text])
        return e[0]

    def build_embeddings(self, documents):
        self.documents = documents
        movies = []
        for d in documents:
            self.document_map[d['id']] = d
            movies.append(f"{d['title']}: {d['description']}")
        self.embeddings = self.model.encode(movies, show_progress_bar=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        movies = []
        for d in documents:
            self.document_map[d['id']] = d
            movies.append(f"{d['title']}: {d['description']}")
        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings=np.load("cache/movie_embeddings.npy")
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        self.embeddings=self.build_embeddings(documents)
        return self.embeddings

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        embedding = self.generate_embedding(query)
        sims = []
        di = 0
        for d in self.embeddings:
            sim = cosine_similarity(embedding, d)
            # we need to put the document in here
            sims.append((sim, self.documents[di]))
            di += 1
        sims.sort(key=lambda x: x[0], reverse=True)
        ret = []
        l = 0
        for s in sims:
            ret.append({"score": s[0], "title": s[1]['title'], "description": s[1]['description']})
            l += 1
            if l >= limit:
                break
        return ret

def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.name}")
    print(f"Max sequence length: {ss.model.max_seq_length}")

def verify_embeddings():
    ss = SemanticSearch()
    with open(datafile, 'r') as f:
        movies = json.load(f)
        documents = movies['movies']
        embeddings = ss.load_or_create_embeddings(documents)
        print(f"Number of docs:   {len(documents)}")
        print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the semantic model")
    subparsers.add_parser("verify_embeddings", help="Verify embeddings")

    embed_parser = subparsers.add_parser("embed_text", help="Get an embedding")
    embed_parser.add_argument("text", type=str, help="Text to embed")

    query_parser = subparsers.add_parser("embedquery", help="Get a query embedding")
    query_parser.add_argument("query", type=str, help="Text to query")

    search_parser = subparsers.add_parser("search", help="Search")
    search_parser.add_argument("query", type=str, help="Text to search")
    search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Number of results")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, nargs='?', default=200, help="Size of a chunk")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "verify_embeddings":
            verify_embeddings()

        case "embed_text":
            embed_text(args.text)

        case "embedquery":
            embed_query_text(args.query)

        case "search":
            ss = SemanticSearch()
            with open(datafile, 'r') as f:
                movies = json.load(f)
                documents = movies['movies']
                ss.load_or_create_embeddings(documents)
                results = ss.search(args.query, args.limit)
                ri = 1
                for r in results:
                    print(f"{ri}. {r['title']} (score: {r['score']})")
                    print(f"   {r['description']}\n")
                    ri += 1

        case "chunk":
            words = args.text.split()
            chunks = []
            chunk = ""
            c = 0
            for w in words:
                if c == 0:
                    chunk = w
                else:
                    chunk = chunk + " " + w
                c += 1

                if c >= args.chunk_size:
                    chunks.append(chunk)
                    chunk = ""
                    c = 0
            if len(chunk) > 0:
                chunks.append(chunk)
            print(f"Chunking {len(args.text)} characters")
            ci = 1
            for ch in chunks:
                print(f"{ci}. {ch}")
                ci += 1

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
