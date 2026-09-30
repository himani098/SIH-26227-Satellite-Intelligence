"""
Phase 6: False-alarm suppression + confidence calibration
SIH Problem Statement 26227 — Ministry of Defence

Takes the raw change candidates from change_detection.py and
down-weights confidence for two known confounders:

  1. Seasonal mismatch — before/after tiles from different seasons
     naturally look different (crop cycles, vegetation) even with no
     real change. Same-season pairs are trusted more.

  2. Image quality — tiles with a high nodata_fraction (from the
     tiling step's cloud/quality filter) are noisier; their change
     scores are trusted less.

Candidates whose ADJUSTED confidence falls below FINAL_THRESHOLD are
kept in the full audit log (nothing is silently deleted — provenance
must be preserved) but excluded from the analyst review queue, per
the "favour precision over indiscriminate recall" requirement.

Usage:
    python false_alarm_suppression.py
    python false_alarm_suppression.py --final_threshold 0.35

Output:
    ../Dataset/Index/review_queue.json               (passes confidence bar)
    ../Dataset/Index/change_candidates_audited.json   (everything, with reasons)
    ../Dataset/analyst_queue.csv                     (tabular queue)
"""

import os
import csv
import json
import argparse
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIH_DIR    = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR).upper() == "CODE" else SCRIPT_DIR
DATASET_DIR= os.path.join(SIH_DIR, "Dataset")
INDEX_DIR  = os.path.join(DATASET_DIR, "Index")

CHANGE_CANDIDATES_FILE = os.path.join(INDEX_DIR, "change_candidates.json")
CATALOGUE_FILE         = os.path.join(DATASET_DIR, "tile_catalogue.csv")

REVIEW_QUEUE_FILE = os.path.join(INDEX_DIR, "review_queue.json")
AUDIT_FILE        = os.path.join(INDEX_DIR, "change_candidates_audited.json")
QUEUE_CSV_FILE    = os.path.join(DATASET_DIR, "analyst_queue.csv")

DEFAULT_FINAL_THRESHOLD = 0.30


def get_season(date_str):
    """Rough season buckets for Jharkhand/Ranchi's climate."""
    month = int(date_str[5:7])
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "summer"
    if month in (6, 7, 8, 9):
        return "monsoon"
    return "post_monsoon"  # 10, 11


def seasonal_penalty(before_date, after_date):
    return 0.0 if get_season(before_date) == get_season(after_date) else 0.30


def quality_factor(nodata_before, nodata_after):
    """1.0 = both tiles clean, lower = noisier tile(s) in the pair."""
    worst = max(nodata_before, nodata_after)
    return 1.0 - worst


def load_nodata_lookup():
    catalogue = pd.read_csv(CATALOGUE_FILE)
    return dict(zip(catalogue["tile_file"], catalogue["nodata_fraction"]))


def calibrate(candidates, nodata_lookup):
    audited = []
    for c in candidates:
        nodata_before = float(nodata_lookup.get(c.get("before_tile", c.get("before_file", "")), 0.0))
        nodata_after  = float(nodata_lookup.get(c.get("after_tile", c.get("after_file", "")), 0.0))

        s_penalty = seasonal_penalty(c["before_date"], c["after_date"])
        q_factor = quality_factor(nodata_before, nodata_after)
        evidence = c.get("false_alarm_evidence", {})
        shadow_fraction = float(evidence.get("shadow_fraction", 0.0))
        shadow_penalty = min(0.25, shadow_fraction * 1.5) if shadow_fraction >= 0.10 else 0.0
        vegetation_delta = abs(float(evidence.get("vegetation_index_delta", 0.0)))
        vegetation_penalty = (
            0.15 if vegetation_delta >= 0.20
            and c.get("change_type") in ("general_change", "water_extent") else 0.0
        )
        timeline = c.get("temporal_timeline", [])
        after_date = c.get("after_date")
        temporary_water = (
            c.get("change_type") == "water_extent"
            and any(item.get("status") == "change_not_detected"
                    and item.get("date", "") > after_date for item in timeline)
        )
        temporary_water_penalty = 0.40 if temporary_water else 0.0

        adjusted_confidence = round(
            float(c["confidence"]) * q_factor * (1 - s_penalty)
            * (1 - shadow_penalty) * (1 - vegetation_penalty)
            * (1 - temporary_water_penalty), 3
        )

        reasons = []
        if s_penalty > 0:
            reasons.append(
                f"seasonal_variation_risk ({get_season(c['before_date'])} vs {get_season(c['after_date'])})"
            )
        if q_factor < 0.9:
            reasons.append(f"low_image_quality (nodata up to {max(nodata_before, nodata_after):.0%})")
            reasons.append("cloud_or_nodata_mask_contamination")
        if shadow_penalty > 0:
            reasons.append(f"possible_shadow_variation ({shadow_fraction:.0%} darkened pixels)")
        if vegetation_penalty > 0:
            reasons.append(f"vegetation_or_seasonal_variation (absolute NDVI delta {vegetation_delta:.2f})")
        if temporary_water:
            reasons.append("temporary_water_candidate (later observation returned toward baseline)")
        if not reasons:
            reasons.append("no_confounders_detected")

        audited.append({
            **c,
            "nodata_before": nodata_before,
            "nodata_after": nodata_after,
            "seasonal_penalty": s_penalty,
            "quality_factor": round(q_factor, 3),
            "shadow_penalty": round(shadow_penalty, 3),
            "vegetation_penalty": round(vegetation_penalty, 3),
            "temporary_water_penalty": round(temporary_water_penalty, 3),
            "adjusted_confidence": adjusted_confidence,
            "suppression_reasons": reasons,
        })

    audited.sort(key=lambda c: c["adjusted_confidence"], reverse=True)
    return audited


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final_threshold", type=float, default=DEFAULT_FINAL_THRESHOLD,
                         help="Minimum adjusted confidence to enter the review queue")
    args = parser.parse_args()

    print("=" * 55)
    print("  SIH-26227 | False-alarm suppression")
    print("=" * 55)

    with open(CHANGE_CANDIDATES_FILE) as f:
        candidates = json.load(f)
    print(f"Loaded {len(candidates)} raw change candidates")

    nodata_lookup = load_nodata_lookup()
    audited = calibrate(candidates, nodata_lookup)

    review_queue = [c for c in audited if c["adjusted_confidence"] >= args.final_threshold]

    with open(AUDIT_FILE, "w") as f:
        json.dump(audited, f, indent=2)
    with open(REVIEW_QUEUE_FILE, "w") as f:
        json.dump(review_queue, f, indent=2)

    # Also save clean analyst queue CSV
    if review_queue:
        keys = ["row", "col", "before_date", "after_date", "change_type",
                "adjusted_confidence", "confidence", "seasonal_penalty", "quality_factor",
            "shadow_penalty", "vegetation_penalty", "temporary_water_penalty",
                "num_objects_marked", "earliest_observation", "preview_file",
                "lat_min", "lon_min", "sensor", "suppression_reasons"]
        csv_rows = []
        for item in review_queue:
            row_dict = {k: item.get(k, "") for k in keys}
            row_dict["suppression_reasons"] = "; ".join(item.get("suppression_reasons", []))
            csv_rows.append(row_dict)
        with open(QUEUE_CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"\nFull audit trail (all candidates) -> {AUDIT_FILE}")
    print(f"Review queue (adjusted_confidence >= {args.final_threshold}) -> {REVIEW_QUEUE_FILE}")
    print(f"Tabular queue -> {QUEUE_CSV_FILE}")
    print(f"\n  Total candidates      : {len(audited)}")
    print(f"  Passed to review queue: {len(review_queue)}")
    print(f"  Suppressed            : {len(audited) - len(review_queue)}")
    print("=" * 55)

    print("\nTop 5 in review queue:")
    for c in review_queue[:5]:
        print(f"  {c['adjusted_confidence']:.3f}  {c['change_type']:25s}  "
              f"{c['before_date']} -> {c['after_date']}  {c['suppression_reasons']}")


if __name__ == "__main__":
    main()
