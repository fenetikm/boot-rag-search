#!/usr/bin/env python3

import argparse
import json
import string
from nltk.stem import PorterStemmer

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
    with open('data/movies.json', 'r') as f:
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

    args = parser.parse_args()
    # test = ["hot", "shot"]
    # for t in test:

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

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
