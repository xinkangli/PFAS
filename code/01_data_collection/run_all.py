"""
Main entry point — runs all collectors, prints stats, exports a CSV.

Usage:
    python run_all.py                       # run all sources
    python run_all.py --sources ecotox nite # run a subset
    python run_all.py --pfas PFOS PFOA      # only certain PFAS
    python run_all.py --export-csv out.csv  # also dump master table to CSV
    python run_all.py --stats-only          # just summarize existing DB
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from schema import init_db, stats
from pfas_targets import by_short, PFAS_TARGETS

from collectors import ecotox, echa, nite_chrip, pubmed_lit


SOURCE_RUNNERS = {
    "ecotox":   ecotox.run,
    "echa":     echa.run,
    "nite":     nite_chrip.run,
    "pubmed":   pubmed_lit.run,   # only fills literature_candidates table
}


def export_csv(conn, path: Path):
    cur = conn.execute("SELECT * FROM pfas_bcf_master")
    cols = [d[0] for d in cur.description]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(cur.fetchall())
    print(f"Exported {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=list(SOURCE_RUNNERS.keys()),
                    choices=list(SOURCE_RUNNERS.keys()))
    ap.add_argument("--pfas", nargs="+", default=None,
                    help="Restrict to certain PFAS shorts, e.g. PFOS PFOA")
    ap.add_argument("--db", default="data/pfas_master.sqlite")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--export-csv", default=None)
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()

    conn = init_db(args.db)

    if args.stats_only:
        print(stats(conn))
        return

    cas_subset = None
    if args.pfas:
        cas_subset = []
        for short in args.pfas:
            m = by_short(short)
            if m:
                cas_subset.append(m["cas"])
            else:
                print(f"WARNING: unknown PFAS short '{short}', skipped")
        if not cas_subset:
            print("No valid PFAS shorts given; aborting.")
            return

    data_dir = Path(args.data_dir)

    for src in args.sources:
        print(f"\n========== source: {src} ==========")
        runner = SOURCE_RUNNERS[src]
        try:
            if src == "pubmed":
                runner(conn, data_dir=data_dir)
            else:
                runner(conn, data_dir=data_dir, cas_subset=cas_subset) \
                    if cas_subset is not None and src != "ecotox" \
                    else runner(conn, data_dir=data_dir) \
                    if src == "ecotox" else \
                    runner(conn, data_dir=data_dir, cas_subset=cas_subset)
        except TypeError:
            # Fallback for runners with different signatures
            runner(conn, data_dir)
        except Exception as e:
            print(f"ERROR running {src}: {e}")

    print("\n========== final stats ==========")
    s = stats(conn)
    print(f"total rows: {s['total_rows']}")
    print(f"by source: {s['by_source']}")
    print(f"by endpoint: {s['by_endpoint']}")
    print(f"by PFAS (top 10):")
    for k, v in list(s["by_pfas"].items())[:10]:
        print(f"  {k:12s} {v}")

    if args.export_csv:
        export_csv(conn, Path(args.export_csv))


if __name__ == "__main__":
    main()
