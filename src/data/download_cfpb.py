"""
Download CFPB consumer complaint data and produce a cleaned JSONL file.

Data source: CFPB Consumer Complaint Database public API
  https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/

Only complaints with a consumer narrative (``complaint_what_happened`` field) are fetched,
filtered to five banking product categories.


Usage
─────
  python -m src.data.download_cfpb
  python -m src.data.download_cfpb --max-records 10000 --output data/processed/complaints.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import Counter
from pathlib import Path

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

API_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"

PRODUCTS = [
    "Credit card or prepaid card",
    "Checking or savings account",
    "Money transfer, virtual currency, or money service",
    "Mortgage",
    "Vehicle loan or lease",
]

PAGE_SIZE = 100
MIN_NARRATIVE_LEN = 50
# The CFPB search API rejects requests with frm >= 10,000 — this is a hard
# Elasticsearch pagination limit, not a bug. Max ~10,000 records per product.
CFPB_MAX_OFFSET = 10_000

# ── HTTP helpers ─────────────────────────────────────────────────────────────


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
async def fetch_page(client: httpx.AsyncClient, params: dict) -> dict:
    """Fetch one page from the CFPB API with automatic retries."""
    response = await client.get(API_URL, params=params, timeout=30.0)
    response.raise_for_status()
    return response.json()


# ── Record processing ────────────────────────────────────────────────────────


def extract_record(hit: dict) -> dict | None:
    """
    Pull the fields we care about from a raw API hit.
    Returns None if the narrative is missing or too short.
    """
    src = hit.get("_source", {})

    narrative: str = src.get("complaint_what_happened") or ""
    if len(narrative.strip()) < MIN_NARRATIVE_LEN:
        return None

    if not src.get("complaint_id"):
        return None

    raw_timely = src.get("timely")
    timely_response = raw_timely == "Yes" if raw_timely is not None else None

    return {
        "complaint_id": src.get("complaint_id"),
        "narrative": narrative.strip(),
        "product": src.get("product"),
        "sub_product": src.get("sub_product"),
        "issue": src.get("issue"),
        "sub_issue": src.get("sub_issue"),
        "company_response": src.get("company_response"),
        "date_received": src.get("date_received"),
        "state": src.get("state"),
        "submitted_via": src.get("submitted_via"),
        "company": src.get("company"),
        "timely_response": timely_response,
    }


# ── Download loop ─────────────────────────────────────────────────────────────


async def download_complaints(
    max_records: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    per_product_limit = max_records // len(PRODUCTS)

    total_downloaded = 0
    total_written = 0
    product_counter: Counter[str] = Counter()
    response_counter: Counter[str] = Counter()
    issue_counter: Counter[str] = Counter()

    async with httpx.AsyncClient() as client:
        with output_path.open("w", encoding="utf-8") as out_file:
            for product in PRODUCTS:
                logger.info(
                    "Fetching product: %s (limit: %d)", product, per_product_limit
                )
                frm = 0
                product_downloaded = 0

                while True:
                    params = {
                        "has_narrative": "true",
                        "product": product,
                        "size": PAGE_SIZE,
                        "frm": frm,
                        "format": "json",
                    }

                    try:
                        data = await fetch_page(client, params)
                    except httpx.HTTPError as exc:
                        logger.error("HTTP error on frm=%d: %s", frm, exc)
                        break

                    hits: list[dict] = data

                    if not hits:
                        break

                    per_product_done = False
                    for hit in hits:
                        total_downloaded += 1
                        product_downloaded += 1
                        record = extract_record(hit)

                        if record is not None:
                            out_file.write(json.dumps(record) + "\n")
                            total_written += 1
                            product_counter[record["product"] or "unknown"] += 1
                            response_counter[record["company_response"] or "unknown"] += 1
                            issue_counter[record["issue"] or "unknown"] += 1

                        if total_downloaded % 1000 == 0:
                            logger.info(
                                "Progress: %d downloaded, %d written",
                                total_downloaded,
                                total_written,
                            )

                        if total_downloaded >= max_records:
                            logger.warning(
                                "Reached overall max-records safety cap (%d).", max_records
                            )
                            _print_summary(
                                total_downloaded, total_written,
                                product_counter, response_counter, issue_counter,
                            )
                            return

                        if product_downloaded >= per_product_limit:
                            logger.info(
                                "Reached per-product limit (%d) for '%s', moving to next product",
                                per_product_limit, product,
                            )
                            per_product_done = True
                            break

                    if per_product_done:
                        break

                    frm += PAGE_SIZE
                    if len(hits) < PAGE_SIZE:
                        break
                    if frm >= CFPB_MAX_OFFSET:
                        logger.warning(
                            "Hit CFPB API pagination limit (frm=%d) for product '%s'. "
                            "This is an API constraint, not a bug. Max ~10,000 records per product.",
                            frm, product,
                        )
                        break

    _print_summary(
        total_downloaded, total_written,
        product_counter, response_counter, issue_counter,
    )


# ── Summary ──────────────────────────────────────────────────────────────────


def _print_summary(
    total_downloaded: int,
    total_written: int,
    product_counter: Counter[str],
    response_counter: Counter[str],
    issue_counter: Counter[str],
) -> None:
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Total records downloaded : {total_downloaded:,}")
    print(f"Records after filtering  : {total_written:,}")
    print()
    print("By product:")
    for product, count in product_counter.most_common():
        print(f"  {product:<50} {count:,}")
    print()
    print("By company_response (raw metadata, not a label):")
    for response, count in response_counter.most_common():
        print(f"  {response:<45} {count:,}")
    print()
    print("By issue (top 15):")
    for issue, count in issue_counter.most_common(15):
        print(f"  {issue:<55} {count:,}")
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CFPB complaint narratives and write to JSONL."
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=100_000,
        help="Stop after this many raw records (pre-filter). Default: 100000.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/complaints.jsonl"),
        help="Destination JSONL file. Default: data/processed/complaints.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    logger.info("Starting download — max_records=%d, output=%s", args.max_records, args.output)
    asyncio.run(download_complaints(max_records=args.max_records, output_path=args.output))


if __name__ == "__main__":
    main()
