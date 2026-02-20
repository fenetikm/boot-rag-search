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

class InvertedIndex:
    index = dict()
    docmap = dict()
    term_frequencies = dict()

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

    def get_documents(self, term):
        if term.lower() not in self.index:
            return []
        return sorted(self.index[term.lower()])

    def get_tf(self, doc_id, term):
        if int(doc_id) not in self.term_frequencies:
            return 0
        return self.term_frequencies[int(doc_id)][term]

    def get_term_doc_count(self, term):
        count = 0
        for d in self.docmap:
            if self.term_frequencies[int(d)][term] > 0:
                count += 1
        return count

    def build(self):
        """
        Build the index.
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

    def load(self):
        with open('cache/index.pkl', 'rb') as i:
            self.index = pickle.load(i)
        with open('cache/docmap.pkl', 'rb') as d:
            self.docmap = pickle.load(d)
        with open('cache/term_frequencies.pkl', 'rb') as t:
            self.term_frequencies = pickle.load(t)

def clean(keyword):
    parts = []
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

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
