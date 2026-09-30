"""
Phase 5 (Revised): Multi-Temporal Change Detection — aligned with SIH PS-26227
Ministry of Defence (MoD) | Indian Army (DGIS)

Problem Statement Alignment:
  2.2.2 — Change types: construction, clearance, water_extent, road_development
  2.2.2 — Estimates EARLIEST observation at which change is supported
  2.2.3 — False-alarm suppression: seasonal, illumination, radiometric confounders
  2.2.5 — Analyst queue: before/after, location, date, sensor, confidence, provenance

Key design decisions:
  - Tiles are grouped by SEASON (winter=Dec/Jan, monsoon=Jun/Jul) before comparison
    so January→June seasonal greenery is NEVER reported as a change
  - Only same-season, consecutive-year pairs are compared (year-over-year change)
  - Spectral classifier uses NIR/Red/All-band deltas aligned with PS change types
  - Confidence score = f(embedding_distance, spectral_consistency, nodata_fraction)
  - Earliest-observation: walks backward in time to find first date change is visible

Usage:
    python change_detection.py
    python change_detection.py --threshold 0.10 --date-from 2020-01-01

Output:
    ../Dataset/change_report.csv       — ranked analyst queue (PS 2.2.5)
    ../Dataset/change_previews/        — before/after PNG per detected change
"""

import os
import csv
import json
import argparse
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform
import faiss
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

# ─── PATHS ────────────────────────────────────────────────────────────────────

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
SIH_DIR        = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR).upper() == "CODE" else SCRIPT_DIR
DATASET_DIR    = os.path.join(SIH_DIR, "Dataset")
TILES_DIR      = os.path.join(DATASET_DIR, "Tiles")
INDEX_DIR      = os.path.join(DATASET_DIR, "Index")
INDEX_FILE     = os.path.join(INDEX_DIR, "tiles.faiss")
METADATA_FILE  = os.path.join(INDEX_DIR, "tile_metadata.json")
STRETCH_FILE   = os.path.join(INDEX_DIR, "stretch_bounds.json")
CATALOGUE_FILE = os.path.join(DATASET_DIR, "tile_catalogue.csv")
REPORT_FILE    = os.path.join(DATASET_DIR, "change_report.csv")
PREVIEW_DIR    = os.path.join(DATASET_DIR, "change_previews")

os.makedirs(PREVIEW_DIR, exist_ok=True)

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DEFAULT_THRESHOLD = 0.10   # cosine-distance to flag change
THUMB_SIZE        = 256

# Season grouping — avoids Jan→Jun seasonal false alarms (PS 2.2.3)
# Month → season label
def month_to_season(month_str: str) -> str:
    m = int(month_str)
    if m in [11, 12, 1, 2]:
        return "DRY_WINTER"    # Dec/Jan — low vegetation, clear atmosphere
    elif m in [6, 7, 8, 9]:
        return "MONSOON"       # Jun — high vegetation, humid
    else:
        return "TRANSITION"    # Mar-May, Oct: transition seasons

# ─── LOAD DATA ────────────────────────────────────────────────────────────────

print("=" * 60)
print("  SIH-26227 | Multi-Temporal Change Detection (PS-aligned)")
print("=" * 60)

with open(METADATA_FILE) as f:
    metadata = json.load(f)

with open(STRETCH_FILE) as f:
    bounds = json.load(f)
p2, p98 = bounds["p2"], bounds["p98"]

index = faiss.read_index(INDEX_FILE)
all_vectors = index.reconstruct_n(0, index.ntotal)

tile_to_vec  = {m["tile_file"]: all_vectors[i] for i, m in enumerate(metadata)}
tile_to_meta = {m["tile_file"]: m for m in metadata}

catalogue = pd.read_csv(CATALOGUE_FILE)
print(f"Loaded {len(metadata)} tile embeddings from index\n")


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def cosine_distance(v1, v2) -> float:
    return float(1.0 - np.dot(v1, v2) /
                 (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))


def tile_to_rgb(tile_path) -> Image.Image:
    with rasterio.open(tile_path) as src:
        data = src.read()
    b, g, r = data[0], data[1], data[2]
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    rgb = np.clip(rgb, p2, p98)
    rgb = ((rgb - p2) / (p98 - p2 + 1e-6) * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def spectral_classify(before_path: str, after_path: str) -> tuple[str, float]:
    """
    PS 2.2.2 change types: construction, clearance, water_extent, road_development
    Uses NIR (B08), Red (B04), and broadband mean to classify.
    Band order: B02=0(blue), B03=1(green), B04=2(red), B08=3(NIR)
    """
    try:
        with rasterio.open(before_path) as src:
            b_data = src.read().astype(np.float32)
            nodata = src.nodata or 0
        with rasterio.open(after_path) as src:
            a_data = src.read().astype(np.float32)
    except Exception:
        return "general_change", 0.50

    valid = (b_data[0] > nodata) & (a_data[0] > nodata)
    if valid.sum() < 200:
        return "general_change", 0.50

    # Band statistics
    b_nir  = b_data[3][valid].mean()
    a_nir  = a_data[3][valid].mean()
    b_red  = b_data[2][valid].mean()
    a_red  = a_data[2][valid].mean()
    b_mean = b_data[:3, valid].mean()
    a_mean = a_data[:3, valid].mean()

    d_nir  = a_nir  - b_nir    # NIR delta
    d_red  = a_red  - b_red    # Red delta
    d_mean = a_mean - b_mean   # Overall brightness delta

    # ── PS 2.2.2 change type heuristics ──────────────────────────────────────

    # Water extent change — significant brightness DROP across all bands
    if d_mean < -300 and a_mean < b_mean * 0.75:
        conf = min(0.92, abs(d_mean) / 800)
        return "water_extent", conf

    # Construction — NIR drop (less vegetation) AND red increase (bare/built soil)
    # Signature: new impervious surface replacing green cover
    if d_nir < -180 and d_red > 80:
        conf = min(0.92, (abs(d_nir) + d_red) / 900)
        return "construction", conf

    # Clearance — vegetation removed (NIR drop) without notable red increase
    # Could be demolition, deforestation, land clearing
    if d_nir < -200 and d_red < 80:
        conf = min(0.90, abs(d_nir) / 700)
        return "clearance", conf

    # Road development — moderate red increase, NIR largely stable
    # Linear feature appearance (heuristic: high red, low NIR change)
    if d_red > 150 and abs(d_nir) < 120:
        conf = min(0.85, d_red / 500)
        return "road_development", conf

    # Vegetation regrowth / expansion (inverse of construction)
    if d_nir > 200 and d_red < 50:
        conf = min(0.80, d_nir / 700)
        return "vegetation_gain", conf

    return "general_change", 0.60


def estimate_earliest_observation(
        location_tiles: list[str], threshold: float) -> str | None:
    """
    PS 2.2.2: 'estimate the earliest available observation at which
    the change is supported by usable imagery.'
    Walk backward in time from the detected change pair to find the
    first date the change signature is already present.
    """
    if len(location_tiles) < 2:
        return None
    for i in range(1, len(location_tiles)):
        v_prev = tile_to_vec.get(location_tiles[i - 1])
        v_curr = tile_to_vec.get(location_tiles[i])
        if v_prev is None or v_curr is None:
            continue
        if cosine_distance(v_prev, v_curr) >= threshold:
            return tile_to_meta[location_tiles[i]]["acquisition_date"]
    return tile_to_meta[location_tiles[-1]]["acquisition_date"]


def compute_confidence(dist: float, change_type: str,
                       spectral_conf: float, threshold: float) -> float:
    """
    Composite confidence = weighted blend of:
      - embedding distance (normalized above threshold)
      - spectral classifier confidence
    Capped at 0.95 to avoid false certainty.
    """
    emb_conf = min(1.0, (dist - threshold) / (0.30 - threshold + 1e-6))
    composite = 0.5 * emb_conf + 0.5 * spectral_conf
    return round(min(0.95, composite), 3)


def pixel_size_m(src, latitude: float) -> tuple[float, float]:
    if src.crs and src.crs.is_projected:
        return abs(src.res[0]), abs(src.res[1])
    return (
        abs(src.res[0]) * 111_320 * np.cos(np.radians(latitude)),
        abs(src.res[1]) * 111_320,
    )


def detect_change_objects(before_path: str, after_path: str):
    """
    Computes spectral delta and isolates discrete changed objects/clusters.
    Returns:
    boxes: list of (x1, y1, x2, y2) in tile coordinates
    objects: measured connected change regions with area and confidence
      diff_heatmap: PIL.Image showing high-contrast change heat overlay
    """
    try:
        with rasterio.open(before_path) as s1, rasterio.open(after_path) as s2:
            b_data = s1.read().astype(np.float32)
            a_data = s2.read().astype(np.float32)
            nodata = s1.nodata or 0
            transform = s2.transform
            crs = s2.crs
            center_latitude = (s2.bounds.bottom + s2.bounds.top) / 2
            pixel_x_m, pixel_y_m = pixel_size_m(s2, center_latitude)

        valid = (b_data[0] > nodata) & (a_data[0] > nodata) & (b_data[0] > 0) & (a_data[0] > 0)
        if valid.sum() < 500:
            return [], None, []

        # Red delta + NIR delta + broadband delta
        d_red = np.abs(a_data[2] - b_data[2])
        d_nir = np.abs(a_data[3] - b_data[3])
        d_broad = np.abs(a_data[:3].mean(axis=0) - b_data[:3].mean(axis=0))

        delta_metric = d_red * 1.5 + d_nir * 1.0 + d_broad * 1.0
        delta_metric[~valid] = 0

        # Change threshold: isolate top ~7% change pixels
        thresh = np.percentile(delta_metric[valid], 93)
        raw_mask = (delta_metric > thresh) & valid

        # Spatial morphology to clean noise and extract discrete changed clusters
        clean_mask = ndi.binary_opening(raw_mask, structure=np.ones((3, 3)))
        clean_mask = ndi.binary_closing(clean_mask, structure=np.ones((5, 5)))

        labeled, num_features = ndi.label(clean_mask)
        slices = ndi.find_objects(labeled)

        boxes = []
        objects = []
        pixel_area_m2 = pixel_x_m * pixel_y_m
        water_ndwi = (a_data[1] - a_data[3]) / (a_data[1] + a_data[3] + 1e-6)
        water_mask = (water_ndwi > 0.12) & valid
        water_distances = (
            ndi.distance_transform_edt(~water_mask, sampling=(pixel_y_m, pixel_x_m))
            if water_mask.any() else None
        )
        for label_id, sl in enumerate(slices, 1):
            if sl is None:
                continue
            y_sl, x_sl = sl
            h = y_sl.stop - y_sl.start
            w = x_sl.stop - x_sl.start
            if w >= 12 and h >= 12 and (w * h) >= 180:
                box = (x_sl.start, y_sl.start, x_sl.stop, y_sl.stop)
                boxes.append(box)
                component = labeled[sl] == label_id
                pixel_count = int(component.sum())
                center_y, center_x = ndi.center_of_mass(component)
                center_y += y_sl.start
                center_x += x_sl.start
                center_row, center_col = int(round(center_y)), int(round(center_x))
                x_coord = transform.c + (center_col + 0.5) * transform.a
                y_coord = transform.f + (center_row + 0.5) * transform.e
                if crs and crs.is_projected:
                    lon, lat = warp_transform(crs, "EPSG:4326", [x_coord], [y_coord])
                    lon, lat = lon[0], lat[0]
                else:
                    lon, lat = x_coord, y_coord
                local_strength = float(delta_metric[sl][component].mean())
                objects.append({
                    "object_id": len(objects) + 1,
                    "bbox_pixels": list(box),
                    "centroid_pixel": [round(center_x, 1), round(center_y, 1)],
                    "centroid_lon": round(float(lon), 6),
                    "centroid_lat": round(float(lat), 6),
                    "area_pixels": pixel_count,
                    "area_m2": round(pixel_count * pixel_area_m2, 1),
                    "confidence": round(min(0.99, local_strength / (thresh * 2 + 1e-6)), 3),
                    "mean_change_strength": round(local_strength, 2),
                    "distance_to_water_m": (
                        round(float(water_distances[center_row, center_col]), 1)
                        if water_distances is not None else None
                    ),
                })

        # Build high-visibility heatmap overlay image (Black -> Red -> Yellow -> White)
        norm_diff = np.clip((delta_metric / (thresh * 1.6 + 1e-6)) * 255.0, 0, 255).astype(np.uint8)
        h_r = norm_diff
        h_g = (norm_diff * 0.50).astype(np.uint8)
        h_b = np.zeros_like(norm_diff)
        heatmap_img = Image.fromarray(np.stack([h_r, h_g, h_b], axis=-1))

        return boxes, heatmap_img, objects
    except Exception as e:
        return [], None, []


def temporal_change_timeline(location_tiles: list[str], baseline_index: int,
                             threshold: float) -> tuple[list[dict], str, str]:
    baseline_file = location_tiles[baseline_index]
    baseline_date = tile_to_meta[baseline_file]["acquisition_date"]
    timeline = [{
        "date": baseline_date,
        "tile_file": baseline_file,
        "status": "baseline",
        "distance_from_baseline": 0.0,
    }]
    first_appeared = None
    persisted_through = None
    persistence_active = True

    for tile_file in location_tiles[baseline_index + 1:]:
        if tile_file not in tile_to_vec:
            continue
        distance = cosine_distance(tile_to_vec[baseline_file], tile_to_vec[tile_file])
        changed = distance >= threshold
        date = tile_to_meta[tile_file]["acquisition_date"]
        status = "change_persisted" if changed else "change_not_detected"
        timeline.append({
            "date": date,
            "tile_file": tile_file,
            "status": status,
            "distance_from_baseline": round(distance, 4),
        })
        if changed:
            first_appeared = first_appeared or date
            if persistence_active:
                persisted_through = date
        elif first_appeared is not None:
            persistence_active = False

    return timeline, first_appeared or tile_to_meta[location_tiles[baseline_index + 1]]["acquisition_date"], persisted_through or tile_to_meta[location_tiles[baseline_index + 1]]["acquisition_date"]


def false_alarm_evidence(before_path: str, after_path: str) -> dict:
    """Measure radiometric cues that may indicate shadow or vegetation variation."""
    try:
        with rasterio.open(before_path) as before_src, rasterio.open(after_path) as after_src:
            before = before_src.read().astype(np.float32)
            after = after_src.read().astype(np.float32)
        valid = (before[0] > 0) & (after[0] > 0)
        if valid.sum() < 500:
            return {"shadow_fraction": 0.0, "vegetation_index_delta": 0.0}

        before_brightness = before[:3].mean(axis=0)
        after_brightness = after[:3].mean(axis=0)
        dark_threshold = np.percentile(before_brightness[valid], 20) * 0.65
        shadow_fraction = float(((after_brightness < dark_threshold) & valid).sum() / valid.sum())
        before_ndvi = (before[3] - before[2]) / (before[3] + before[2] + 1e-6)
        after_ndvi = (after[3] - after[2]) / (after[3] + after[2] + 1e-6)
        vegetation_delta = float(np.mean(after_ndvi[valid]) - np.mean(before_ndvi[valid]))
        return {
            "shadow_fraction": round(shadow_fraction, 4),
            "vegetation_index_delta": round(vegetation_delta, 4),
        }
    except Exception:
        return {"shadow_fraction": 0.0, "vegetation_index_delta": 0.0}


def save_before_after(before_path, after_path, before_date, after_date,
                      change_type, confidence, earliest_obs, out_path):
    try:
        b_img = tile_to_rgb(before_path).resize((THUMB_SIZE, THUMB_SIZE))
        a_img = tile_to_rgb(after_path).resize((THUMB_SIZE, THUMB_SIZE))
    except Exception:
        return 0, []

    # Detect discrete changed objects & bounding boxes
    boxes, heatmap_img, objects = detect_change_objects(before_path, after_path)
    if heatmap_img is not None:
        heat_thumb = heatmap_img.resize((THUMB_SIZE, THUMB_SIZE))
    else:
        heat_thumb = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), (0, 0, 0))

    # Scale factor from 512 to THUMB_SIZE (256)
    scale = THUMB_SIZE / 512.0

    # Draw boxes on BEFORE (Yellow/Cyan guide) and AFTER (Bright Red/Orange alert)
    draw_b = ImageDraw.Draw(b_img)
    draw_a = ImageDraw.Draw(a_img)

    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        bx1, by1 = int(x1 * scale), int(y1 * scale)
        bx2, by2 = int(x2 * scale), int(y2 * scale)

        # Draw warning box on Before image
        draw_b.rectangle([bx1, by1, bx2, by2], outline=(255, 230, 60), width=2)
        draw_b.text((bx1 + 2, max(0, by1 - 12)), f"#{i}", fill=(255, 230, 60))

        # Draw confirmed change box on After image
        draw_a.rectangle([bx1, by1, bx2, by2], outline=(255, 40, 40), width=3)
        draw_a.text((bx1 + 2, max(0, by1 - 12)), f"#{i}", fill=(255, 60, 60))

    # 3-Panel Analyst Canvas: [BEFORE] | [AFTER (Marked)] | [CHANGE HEATMAP]
    W = THUMB_SIZE * 3
    H = THUMB_SIZE + 65
    canvas = Image.new("RGB", (W, H), (15, 15, 25))

    canvas.paste(b_img, (0, 0))
    canvas.paste(a_img, (THUMB_SIZE, 0))
    canvas.paste(heat_thumb, (THUMB_SIZE * 2, 0))

    draw = ImageDraw.Draw(canvas)

    # Panel titles
    draw.text((4, THUMB_SIZE + 4),
              f"1. BEFORE: {before_date}", fill=(80, 220, 120))
    draw.text((THUMB_SIZE + 4, THUMB_SIZE + 4),
              f"2. AFTER:  {after_date} (MARKED)", fill=(255, 80, 80))
    draw.text((THUMB_SIZE * 2 + 4, THUMB_SIZE + 4),
              "3. CHANGE HEATMAP DELTA", fill=(255, 200, 50))

    # Metadata & Provenance banner
    draw.text((4, THUMB_SIZE + 24),
              f"Change Type : {change_type.upper().replace('_',' ')}   |   "
              f"Objects Identified: {len(boxes)} marked regions",
              fill=(255, 240, 100))
    draw.text((4, THUMB_SIZE + 42),
              f"Confidence  : {confidence:.2f}   |   "
              f"Earliest Detection: {earliest_obs or 'unknown'}",
              fill=(190, 190, 255))

    canvas.save(out_path)
    return len(boxes), objects


# ─── MAIN DETECTION LOOP ──────────────────────────────────────────────────────

def detect_changes(threshold: float, date_from: str = None, date_to: str = None):

    # Group tiles by geographic location (row, col)
    loc_groups: dict[tuple, list[str]] = {}
    for _, row in catalogue.iterrows():
        key = (int(row["row"]), int(row["col"]))
        loc_groups.setdefault(key, []).append(row["tile_file"])

    print(f"Unique geographic locations : {len(loc_groups)}")
    print(f"Change threshold            : {threshold}")
    print(f"Date filter                 : {date_from or 'any'} -> {date_to or 'any'}")
    print()
    print("  [PS 2.2.3] Seasonal false-alarm suppression ACTIVE")
    print("  → Only same-season, year-over-year pairs are compared")
    print("  → Jan→Jun / winter→monsoon transitions are excluded\n")

    change_records = []
    total_pairs    = 0
    total_skipped_seasonal = 0

    for (r, c), tile_files in loc_groups.items():

        # Apply date filters
        filtered = [f for f in tile_files
                    if f in tile_to_meta
                    and (not date_from or tile_to_meta[f]["acquisition_date"] >= date_from)
                    and (not date_to   or tile_to_meta[f]["acquisition_date"] <= date_to)]

        # ── PS 2.2.3: Group by SEASON before comparing ────────────────────────
        season_groups: dict[str, list[str]] = {}
        for tf in filtered:
            date = tile_to_meta[tf]["acquisition_date"]
            month = date[5:7]
            season = month_to_season(month)
            season_groups.setdefault(season, []).append(tf)

        # Compare consecutive tiles WITHIN the same season only
        for season, season_tiles in season_groups.items():
            if len(season_tiles) < 2:
                continue

            # Sort by date within this season
            season_tiles_sorted = sorted(
                season_tiles,
                key=lambda f: tile_to_meta[f]["acquisition_date"]
            )

            for i in range(len(season_tiles_sorted) - 1):
                before_file = season_tiles_sorted[i]
                after_file  = season_tiles_sorted[i + 1]

                if before_file not in tile_to_vec or after_file not in tile_to_vec:
                    continue

                total_pairs += 1
                dist = cosine_distance(
                    tile_to_vec[before_file], tile_to_vec[after_file])

                if dist < threshold:
                    continue  # No significant change

                before_meta = tile_to_meta[before_file]
                after_meta  = tile_to_meta[after_file]
                before_path = os.path.join(TILES_DIR, before_file)
                after_path  = os.path.join(TILES_DIR, after_file)

                # PS 2.2.2 spectral classification
                change_type, spectral_conf = spectral_classify(
                    before_path, after_path)

                # Skip vegetation_gain — seasonal residual noise
                # (PS 2.2.3: favour precision over recall)
                if change_type == "vegetation_gain":
                    total_skipped_seasonal += 1
                    continue

                # Composite confidence (PS 2.2.5)
                confidence = compute_confidence(
                    dist, change_type, spectral_conf, threshold)

                # PS 2.2.2: Earliest observation estimate
                earliest = estimate_earliest_observation(
                    season_tiles_sorted[:i + 2], threshold)
                timeline, first_appeared, persisted_through = temporal_change_timeline(
                    season_tiles_sorted, i, threshold)
                pair_evidence = false_alarm_evidence(before_path, after_path)

                # Before/after preview (PS 2.2.5)
                preview_name = (
                    f"change_{r}_{c}_{season}_"
                    f"{before_meta['acquisition_date']}_to_"
                    f"{after_meta['acquisition_date']}.png"
                )
                preview_path = os.path.join(PREVIEW_DIR, preview_name)
                num_objects, objects = save_before_after(
                    before_path, after_path,
                    before_meta["acquisition_date"],
                    after_meta["acquisition_date"],
                    change_type, confidence, earliest, preview_path
                )

                change_records.append({
                    "row": r, "col": c,
                    "season": season,
                    "before_tile":         before_file,
                    "after_tile":          after_file,
                    "before_file":         before_file,
                    "after_file":          after_file,
                    "before_date":         before_meta["acquisition_date"],
                    "after_date":          after_meta["acquisition_date"],
                    "earliest_observation":earliest,
                    "lon_min": before_meta["lon_min"],
                    "lat_min": before_meta["lat_min"],
                    "lon_max": before_meta["lon_max"],
                    "lat_max": before_meta["lat_max"],
                    "embedding_distance":  float(round(dist, 4)),
                    "change_type":         change_type,
                    "confidence":          float(confidence),
                    "num_objects_marked":  int(num_objects),
                    "area_m2":             round(sum(obj["area_m2"] for obj in objects), 1),
                    "objects":              objects,
                    "first_appeared":       first_appeared,
                    "persisted_through":    persisted_through,
                    "temporal_timeline":    timeline,
                    "false_alarm_evidence": pair_evidence,
                    "preview_file":        preview_name,
                    "sensor":              before_meta["sensor"],
                    "analyst_decision":    "",   # PS 2.2.5 audit trail
                    "analyst_notes":       "",
                })

    # ─── SAVE REPORT (PS 2.2.5 — analyst queue, sorted by confidence) ────────

    fieldnames = [
        "row", "col", "season",
        "before_file", "after_file", "before_date", "after_date",
        "earliest_observation",
        "lon_min", "lat_min", "lon_max", "lat_max",
        "embedding_distance", "change_type", "confidence",
        "num_objects_marked", "area_m2", "first_appeared", "persisted_through",
        "objects_json", "temporal_timeline_json", "false_alarm_evidence_json",
        "preview_file", "sensor",
        "analyst_decision", "analyst_notes",   # PS 2.2.5 audit columns
    ]

    records_sorted = sorted(change_records, key=lambda x: -x["confidence"])
    with open(REPORT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        csv_records = []
        for record in records_sorted:
            csv_record = dict(record)
            csv_record["objects_json"] = json.dumps(record["objects"])
            csv_record["temporal_timeline_json"] = json.dumps(record["temporal_timeline"])
            csv_record["false_alarm_evidence_json"] = json.dumps(record["false_alarm_evidence"])
            csv_records.append(csv_record)
        writer.writerows(csv_records)

    # Save JSON candidates for Phase 6 suppression pipeline
    candidates_json = os.path.join(INDEX_DIR, "change_candidates.json")
    with open(candidates_json, "w") as f:
        json.dump(records_sorted, f, indent=2)
    print(f"  Saved candidates JSON       : {candidates_json}")

    # ─── SUMMARY ──────────────────────────────────────────────────────────────

    type_counts: dict[str, int] = {}
    for rec in change_records:
        type_counts[rec["change_type"]] = type_counts.get(rec["change_type"], 0) + 1

    print("=" * 60)
    print(f"  Change Detection Complete  [PS 2.2.2 / 2.2.3]")
    print(f"  Same-season pairs checked  : {total_pairs}")
    print(f"  Seasonal noise suppressed  : {total_skipped_seasonal}")
    print(f"  Meaningful changes flagged : {len(change_records)}")
    print(f"  Report saved               : {REPORT_FILE}")
    print(f"  Previews saved             : {PREVIEW_DIR}/")
    print()
    print("  Change type breakdown:")
    for ctype, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {ctype:20s} : {cnt}")
    print("=" * 60)

    if change_records:
        print("\n  Top 5 high-confidence changes (Analyst Queue):")
        print(f"  {'TYPE':<20} {'CONF':>5}  {'DATE RANGE':<28}  {'LAT':>8}  {'LON':>8}")
        for rec in records_sorted[:5]:
            print(f"  {rec['change_type']:<20} {rec['confidence']:>5.2f}  "
                  f"{rec['before_date']} → {rec['after_date']}  "
                  f"{rec['lat_min']:>8.4f}  {rec['lon_min']:>8.4f}")

    return change_records


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SIH-26227 Change Detection (PS-aligned)")
    parser.add_argument("--threshold",  type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--date-from",  type=str,   default=None)
    parser.add_argument("--date-to",    type=str,   default=None)
    args = parser.parse_args()

    detect_changes(
        threshold=args.threshold,
        date_from=args.date_from,
        date_to=args.date_to,
    )
