import argparse
import logging

import pandas as pd

from src.retrieval.bm25_index import BM25Index

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/complaints_deduped.jsonl")
    parser.add_argument("--output", default="data/processed/bm25_index.pkl")
    args = parser.parse_args()

    df = pd.read_json(args.input, lines=True)
    df = df.dropna(subset=["narrative"])

    logger.info("Building BM25 index over %d narratives", len(df))
    index = BM25Index.build(
        ids=df["complaint_id"].astype(str).tolist(),
        texts=df["narrative"].astype(str).tolist(),
    )
    index.save(args.output)
    logger.info("Saved BM25 index to %s", args.output)


if __name__ == "__main__":
    main()
