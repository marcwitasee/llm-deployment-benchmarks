"""Pull triaged Firefox/Core defects from Mozilla Bugzilla into bronze JSONL.

Run on your Mac (my container can't reach Bugzilla):
    python pull_bugzilla.py --max 200                      # small sample to eyeball
    python pull_bugzilla.py --per-month 400 --since 2024-01-01 --until 2026-01-01

--per-month spreads the pull across every month in the range. Note: it keeps
the FIRST N bugs of each month, so it only covers each month's first days.
For the dashboards use --all-metadata, which takes every bug in the range
but skips descriptions (fast: one request per 1,000 bugs):
    python pull_bugzilla.py --all-metadata --since 2024-01-01 --until 2026-01-01 --out bugzilla_meta_full.jsonl
 Bronze keeps everything as pulled;
automated test-failure bugs get filtered out later, in silver.

Output: bugzilla_bronze.jsonl, one bug per line, with the first comment
(the text as filed) attached as "description".
"""
import argparse
import json
import time

import requests

API = "https://bugzilla.mozilla.org/rest"
FIELDS = ("id,summary,product,component,severity,priority,type,status,resolution,"
          "creation_time,cf_last_resolved,last_change_time")


def get(url, params):
    # Anonymous reads are rate-limited per IP: back off and retry.
    for attempt in range(6):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        wait = 2 ** attempt
        print(f"  HTTP {r.status_code}, retrying in {wait}s")
        time.sleep(wait)
    r.raise_for_status()


def search(since, max_bugs, until=None, page=1000):
    params = {
        "product": ["Firefox", "Core"],
        "type": "defect",
        "severity": ["S1", "S2", "S3", "S4"],   # triaged only
        "creation_time": since,
        "include_fields": FIELDS,
        "order": "bug_id",
        "limit": page,
    }
    if until:  # created strictly before this date
        params.update({"f1": "creation_ts", "o1": "lessthan", "v1": until})
    bugs, offset = [], 0
    while len(bugs) < max_bugs:
        batch = get(f"{API}/bug", {**params, "offset": offset})["bugs"]
        bugs.extend(batch)
        print(f"searched {len(bugs)} bugs")
        if len(batch) < page:
            break
        offset += page
    return bugs[:max_bugs]


def months(since, until):
    y, m = map(int, since[:7].split("-"))
    end = until[:7]
    while f"{y:04d}-{m:02d}" < end:
        start = f"{y:04d}-{m:02d}-01"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        yield start, f"{y:04d}-{m:02d}-01"


def first_comment(bug_id):
    data = get(f"{API}/bug/{bug_id}/comment", {"include_fields": "text,count"})
    comments = data["bugs"][str(bug_id)]["comments"]
    return comments[0]["text"] if comments else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--since", default="2024-01-01")
    ap.add_argument("--until", default=None, help="needed with --per-month")
    ap.add_argument("--per-month", type=int, default=0,
                    help="cap per calendar month; 0 = one pull of --max bugs")
    ap.add_argument("--all-metadata", action="store_true",
                    help="every bug month by month, no descriptions")
    ap.add_argument("--out", default="bugzilla_bronze.jsonl")
    args = ap.parse_args()

    if args.all_metadata:
        with open(args.out, "w") as f:
            total = 0
            for start, end in months(args.since, args.until):
                month = search(start, 10**9, until=end)
                for bug in month:
                    f.write(json.dumps(bug) + "\n")
                total += len(month)
                print(f"== {start[:7]}: {len(month)} bugs (total {total})")
                if len(month) >= 10000:
                    print("   WARNING: hit the 10,000 search cap; this month is incomplete")
        print(f"wrote {total} bugs to {args.out}")
        return

    if args.per_month:
        bugs = []
        for start, end in months(args.since, args.until):
            print(f"== {start[:7]}")
            bugs.extend(search(start, args.per_month, until=end))
    else:
        bugs = search(args.since, args.max)
    with open(args.out, "w") as f:
        for i, bug in enumerate(bugs, 1):
            bug["description"] = first_comment(bug["id"])
            f.write(json.dumps(bug) + "\n")
            if i % 100 == 0:
                print(f"fetched descriptions for {i}/{len(bugs)}")
            time.sleep(0.2)
    print(f"wrote {len(bugs)} bugs to {args.out}")


if __name__ == "__main__":
    main()
