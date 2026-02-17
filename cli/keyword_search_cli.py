#!/usr/bin/env python3

import argparse
import json
from re import split
import string

def search(keyword):
    matches = []
    clean_keyword = keyword.translate(str.maketrans('', '', string.punctuation)).lower()
    parts = clean_keyword.split(' ')
    if len(parts) == 0:
        return matches
    with open('data/movies.json') as f:
        movies = json.load(f)
        for m in movies['movies']:
            for p in parts:
                if p in m['title'].translate(str.maketrans('', '', string.punctuation)).lower():
                    matches.append(m['title'])
    return matches

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

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

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
