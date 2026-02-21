#!/usr/bin/env python3

import argparse
import json
import string
import os
import pickle
from typing import Counter
from nltk.stem import PorterStemmer
import math

datafile = 'data/movies.json'
# datafile = 'data/killshot.json'
# datafile = 'data/killshot2.json'
# datafile = 'data/movie1.json'

BM25_K1 = 1.5
BM25_B = 0.75

class InvertedIndex:
    def __init__(self) -> None:
        self.index = dict()
        self.docmap = dict()
        self.term_frequencies = dict()
        self.doc_lengths = dict()

    def __add_document(self, doc_id, text):
        tokens = clean(text)
        for t in tokens:
            if t not in self.index:
                self.index[t] = {doc_id}
            else:
                self.index[t].add(doc_id)
            if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = Counter([t])
            else:
                self.term_frequencies[doc_id].update([t])
        self.doc_lengths[doc_id] = len(tokens)

    def get_documents(self, term):
        if term.lower() not in self.index:
            return []
        return sorted(self.index[term.lower()])

    def get_tf(self, doc_id, term):
        if int(doc_id) not in self.term_frequencies:
            print("not found")
            return 0
        return self.term_frequencies[int(doc_id)][term]

    def get_term_doc_count(self, term):
        count = 0
        for d in self.docmap:
            if self.term_frequencies[int(d)][term] > 0:
                count += 1
        return count

    def get_bm25_idf(self, term: str) -> float:
        clean_term = clean(term)
        if len(clean_term) != 1:
            raise Exception("Bad term")
        term1 = clean_term[0]
        n = len(self.docmap)
        df = len(self.get_documents(term1))
        bm25 = math.log((n - df + 0.5) / (df + 0.5) + 1)
        return bm25

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        clean_term = clean(term)
        if len(clean_term) != 1:
            raise Exception("Bad term")
        term1 = clean_term[0]
        tf = self.get_tf(doc_id, term1)
        b = BM25_B
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        sat_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return sat_tf

    def build(self):
        """
        Build the index.
        Example of a method comment!
        """
        with open(datafile, 'r') as f:
            movies = json.load(f)

            id = 1
            for m in movies['movies']:
                self.__add_document(id, f"{m['title']} {m['description']}")
                self.docmap[id] = m
                id += 1

    def save(self):
        if not os.path.isdir('cache'):
            os.makedirs('cache')
        with open('cache/index.pkl', 'wb') as i:
            pickle.dump(self.index, i, protocol=pickle.HIGHEST_PROTOCOL)
        with open('cache/docmap.pkl', 'wb') as d:
            pickle.dump(self.docmap, d, protocol=pickle.HIGHEST_PROTOCOL)
        with open('cache/term_frequencies.pkl', 'wb') as t:
            pickle.dump(self.term_frequencies, t, protocol=pickle.HIGHEST_PROTOCOL)
        with open('cache/doc_lengths.pkl', 'wb') as l:
            pickle.dump(self.doc_lengths, l, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open('cache/index.pkl', 'rb') as i:
            self.index = pickle.load(i)
        with open('cache/docmap.pkl', 'rb') as d:
            self.docmap = pickle.load(d)
        with open('cache/term_frequencies.pkl', 'rb') as t:
            self.term_frequencies = pickle.load(t)
        with open('cache/doc_lengths.pkl', 'rb') as l:
            self.doc_lengths = pickle.load(l)

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        total = 0.0
        for l in self.doc_lengths:
            total += self.doc_lengths[l]
        return total / len(self.doc_lengths)

# these next two methods come from boot.dev... and they have issues!
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def bad_tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    with open('data/stopwords.txt', 'r') as f:
        content = f.read()
        stop_words = content.splitlines()
    filtered_words = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words

def clean(keyword):
    return bad_tokenize_text(keyword)

def correct_clean(keyword):
    parts = []
    keyword = keyword.replace("\n", " ")
    keyword = keyword.encode('ascii', 'ignore').decode('ascii')
    keyword = keyword.replace("\\u2019", " ")
    clean_keyword = keyword.translate(str.maketrans('', '', string.punctuation)).lower()
    parts = clean_keyword.split(' ')
    if len(parts) == 0:
        return []
    clean_parts = []
    with open('data/stopwords.txt', 'r') as f:
        content = f.read()
        stops = content.splitlines()
    for p in parts:
        if p not in stops:
            stemmer = PorterStemmer()
            clean_parts.append(stemmer.stem(p))
    return clean_parts

def search(keyword):
    max_results = 5
    keyword_parts = clean(keyword)
    if len(keyword_parts) == 0:
        return []
    matches = []
    ii = InvertedIndex()
    ii.load()
    count = 0
    for k in keyword_parts:
        if count >= max_results:
            break
        result = ii.get_documents(k)
        for r in result:
            matches.append(f"{ii.docmap[r]['title']} {r}")
            count += 1
            if count >= max_results:
                break
    return matches

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build inverted index")
    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("doc_id", type=str, help="Document id")
    tf_parser.add_argument("term", type=str, help="Term")

    idf_parser = subparsers.add_parser("idf", help="Calc inverse document frequency")
    idf_parser.add_argument("term", type=str, help="Term")

    tfidf_parser = subparsers.add_parser("tfidf", help="Calc inverse document frequency")
    tfidf_parser.add_argument("doc_id", type=str, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
      "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            matches = search(args.query)
            if len(matches) > 0:
                index = 1
                for m in matches:
                    print((str(index) + "."), m)
                    index += 1
                    if index > 5:
                        break

        case "build":
            ii = InvertedIndex()
            ii.build()
            ii.save()

        case "tf":
            term = clean(args.term)[0]
            ii = InvertedIndex()
            ii.load()
            freq = ii.get_tf(args.doc_id, term)

            print(freq)

        case "idf":
            term = clean(args.term)[0]
            ii = InvertedIndex()
            ii.load()
            total_doc_count = len(ii.docmap)
            term_match_doc_count = ii.get_term_doc_count(term)
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))

            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            term = clean(args.term)[0]
            ii = InvertedIndex()
            ii.load()
            total_doc_count = len(ii.docmap)
            term_match_doc_count = ii.get_term_doc_count(term)
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            tf_idf = idf * ii.get_tf(args.doc_id, term)

            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "bm25idf":
            ii = InvertedIndex()
            ii.load()
            bm25idf = ii.get_bm25_idf(args.term)

            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            ii = InvertedIndex()
            ii.load()
            bm25tf = ii.get_bm25_tf(args.doc_id, args.term, args.k1, args.b)

            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
