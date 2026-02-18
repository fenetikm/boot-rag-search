#!/usr/bin/env python3

import argparse
import json
import string
import os
import pickle
from nltk.stem import PorterStemmer

datafile = 'data/movies.json'
# datafile = 'data/killshot.json'

class InvertedIndex:
    index = dict()
    docmap = dict()
    def __add_document(self, doc_id, text):
        tokens = clean(text)
        for t in tokens:
            if t not in self.index:
                self.index[t] = {doc_id}
            else:
                self.index[t].add(doc_id)
    def get_documents(self, term):
        if term.lower() not in self.index:
            return []
        return sorted(self.index[term.lower()])
    def build(self):
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
    keyword_parts = clean(keyword)
    if len(keyword_parts) == 0:
        return []
    matches = []
    with open(datafile, 'r') as f:
        movies = json.load(f)
        for m in movies['movies']:
            title_parts = clean(m['title'])
            found = False
            for k in keyword_parts:
                if found:
                    break
                for t in title_parts:
                    if found:
                        break
                    if k in t:
                        matches.append(m['title'])
                        found = True
    return matches

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build inverted index")

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
            merida = ii.get_documents('merida')
            print(merida)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
