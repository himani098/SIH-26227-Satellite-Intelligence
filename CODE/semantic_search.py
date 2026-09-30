"""
Phase 4: Semantic & Multimodal Search over the tile index
SIH Problem Statement 26227 — Ministry of Defence

Supports:
  - Text-to-image search : free-text query → ranked tile results
  - Image-to-image search: a tile path → visually similar tiles
  - Metadata filters     : date range, AOI bounding box

Usage:
    python semantic_search.py --query "newly built structures near a river" --top 5
    python semantic_search.py --query "large open ground" --top 5 --date-from 2022-01-01
    python semantic_search.py --image Dataset/Tiles/ranchi_2020_01_tile_2_3.tif --top 5

Output:
    Prints ranked results to console.
    Saves a side-by-side PNG preview of top results → Dataset/search_results/
"""

import os
import csv
import json
import argparse
import numpy as np
import torch
import open_clip
import faiss
import rasterio
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
try:
    from .query_understanding import parse_query
except ImportError:
    from query_understanding import parse_query

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SIH_DIR      = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR).upper() == "CODE" else SCRIPT_DIR
DATASET_DIR  = os.path.join(SIH_DIR, "Dataset")
TILES_DIR    = os.path.join(DATASET_DIR, "Tiles")
INDEX_DIR    = os.path.join(DATASET_DIR, "Index")
INDEX_FILE   = os.path.join(INDEX_DIR, "tiles.faiss")
METADATA_FILE= os.path.join(INDEX_DIR, "tile_metadata.json")
STRETCH_FILE = os.path.join(INDEX_DIR, "stretch_bounds.json")
RESULTS_DIR  = os.path.join(DATASET_DIR, "search_results")

os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── LOAD MODEL + INDEX ───────────────────────────────────────────────────────

print("Loading CLIP model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(DEVICE).eval()

print("Loading FAISS index...")
index = faiss.read_index(INDEX_FILE)

with open(METADATA_FILE) as f:
    metadata = json.load(f)

with open(STRETCH_FILE) as f:
    bounds = json.load(f)
p2, p98 = bounds["p2"], bounds["p98"]

print(f"Index ready — {index.ntotal} tiles loaded.\n")


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def tile_to_rgb(tile_path):
    """Convert 4-band tile to RGB PIL image using saved global stretch bounds."""
    with rasterio.open(tile_path) as src:
        data = src.read()
    blue, green, red = data[0], data[1], data[2]
    rgb = np.stack([red, green, blue], axis=-1).astype(np.float32)
    rgb = np.clip(rgb, p2, p98)
    rgb = ((rgb - p2) / (p98 - p2 + 1e-6) * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def embed_text(query: str) -> np.ndarray:
    tokens = tokenizer([query]).to(DEVICE)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")


def embed_image_tile(tile_path: str) -> np.ndarray:
    img = tile_to_rgb(tile_path)
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")


def apply_filters(results, date_from=None, date_to=None,
                  lon_min=None, lat_min=None, lon_max=None, lat_max=None):
    """Filter metadata results by date range and/or AOI bounding box."""
    filtered = []
    for score, meta in results:
        date = meta["acquisition_date"]
        if date_from and date < date_from:
            continue
        if date_to and date > date_to:
            continue
        if lon_min is not None and meta["lon_max"] < lon_min:
            continue
        if lon_max is not None and meta["lon_min"] > lon_max:
            continue
        if lat_min is not None and meta["lat_max"] < lat_min:
            continue
        if lat_max is not None and meta["lat_min"] > lat_max:
            continue
        filtered.append((score, meta))
    return filtered


def tile_has_water_near_center(tile_path: str, radius_m: float) -> bool:
    """Use a local NDWI mask to enforce tile-center proximity to surface water."""
    try:
        with rasterio.open(tile_path) as src:
            green = src.read(2).astype(np.float32)
            nir = src.read(4).astype(np.float32)
            water = ((green - nir) / (green + nir + 1e-6) > 0.12) & (green > 0)
            if not water.any():
                return False

            if src.crs and src.crs.is_projected:
                pixel_x, pixel_y = abs(src.res[0]), abs(src.res[1])
            else:
                center_lat = (src.bounds.bottom + src.bounds.top) / 2
                pixel_y = abs(src.res[1]) * 111_320
                pixel_x = abs(src.res[0]) * 111_320 * np.cos(np.radians(center_lat))
            distances = ndi.distance_transform_edt(
                ~water, sampling=(pixel_y, pixel_x)
            )
            center = (src.height // 2, src.width // 2)
            return bool(distances[center] <= radius_m)
    except (OSError, ValueError, IndexError):
        return False


def search(query_vector: np.ndarray, top_k: int = 10):
    """Search FAISS index and return (score, metadata) pairs."""
    scores, indices = index.search(query_vector, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append((float(score), metadata[idx]))
    return results


def save_results_preview(results, query_label: str, out_path: str):
    """Save a horizontal strip of top result tile previews as PNG."""
    THUMB = 256
    n = len(results)
    canvas = Image.new("RGB", (THUMB * n, THUMB + 40), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    for i, (score, meta) in enumerate(results):
        tile_path = os.path.join(TILES_DIR, meta["tile_file"])
        if not os.path.exists(tile_path):
            continue
        thumb = tile_to_rgb(tile_path).resize((THUMB, THUMB))
        canvas.paste(thumb, (i * THUMB, 0))
        label = f"{score:.3f} | {meta['acquisition_date']}"
        draw.text((i * THUMB + 4, THUMB + 4), label, fill=(220, 220, 100))

    draw.text((4, THUMB + 22), f"Query: {query_label[:80]}", fill=(180, 180, 180))
    canvas.save(out_path)
    print(f"Preview saved → {out_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SIH-26227 Semantic Tile Search")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query",  type=str, help="Free-text search query")
    group.add_argument("--image", "--image_query", dest="image", type=str,
                       help="Path or filename of a tile for image-to-image search")

    parser.add_argument("--top", "--top_k", dest="top", type=int, default=5,
                        help="Number of results to return")
    parser.add_argument("--date-from", type=str,   default=None, help="Filter: start date (YYYY-MM-DD)")
    parser.add_argument("--date-to",   type=str,   default=None, help="Filter: end date   (YYYY-MM-DD)")
    parser.add_argument("--lon-min",   type=float, default=None, help="Filter: west longitude")
    parser.add_argument("--lon-max",   type=float, default=None, help="Filter: east longitude")
    parser.add_argument("--lat-min",   type=float, default=None, help="Filter: south latitude")
    parser.add_argument("--lat-max",   type=float, default=None, help="Filter: north latitude")

    args = parser.parse_args()

    # Build query embedding
    if args.query:
        print(f'Text query : "{args.query}"')
        query_vec   = embed_text(args.query)
        query_label = args.query
        query_constraints = parse_query(args.query)
    else:
        image_path = args.image
        if not os.path.exists(image_path):
            candidate = os.path.join(TILES_DIR, os.path.basename(image_path))
            if os.path.exists(candidate):
                image_path = candidate
            else:
                print(f"Error: image tile not found: {args.image}")
                return
        print(f"Image query: {image_path}")
        query_vec   = embed_image_tile(image_path)
        query_label = os.path.basename(image_path)
        query_constraints = {}

    # Search (fetch larger pool before filtering so filters don't exhaust results)
    pool_size = min(args.top * 10, index.ntotal)
    raw_results = search(query_vec, top_k=pool_size)

    # Apply metadata filters
    filtered = apply_filters(
        raw_results,
        date_from=args.date_from or query_constraints.get("date_from"),
        date_to=args.date_to or query_constraints.get("date_to"),
        lon_min=args.lon_min if args.lon_min is not None else (query_constraints.get("location_bounds") or {}).get("lon_min"),
        lat_min=args.lat_min if args.lat_min is not None else (query_constraints.get("location_bounds") or {}).get("lat_min"),
        lon_max=args.lon_max if args.lon_max is not None else (query_constraints.get("location_bounds") or {}).get("lon_max"),
        lat_max=args.lat_max if args.lat_max is not None else (query_constraints.get("location_bounds") or {}).get("lat_max"),
    )
    if query_constraints.get("near_feature") == "water":
        filtered = [
            (score, meta) for score, meta in filtered
            if tile_has_water_near_center(
                os.path.join(TILES_DIR, meta["tile_file"]), query_constraints["radius_m"]
            )
        ]
    top_results = filtered[: args.top]

    # Print results
    print(f"\n{'='*55}")
    print(f"  Top {len(top_results)} results")
    print(f"{'='*55}")
    for rank, (score, meta) in enumerate(top_results, 1):
        print(f"\n  #{rank}  Score : {score:.4f}")
        print(f"       File  : {meta['tile_file']}")
        print(f"       Date  : {meta['acquisition_date']}")
        print(f"       Loc   : lat [{meta['lat_min']:.4f}, {meta['lat_max']:.4f}]"
              f"  lon [{meta['lon_min']:.4f}, {meta['lon_max']:.4f}]")
        print(f"       Sensor: {meta['sensor']}")
    print(f"\n{'='*55}")

    # Save a preview when available and always replace the result CSV.
    csv_file = os.path.join(RESULTS_DIR, "search_results.csv")
    csv_fieldnames = ["rank", "query", "object", "location", "near_feature", "radius_m",
                      "score", "tile_file", "acquisition_date", "lat_min", "lat_max",
                      "lon_min", "lon_max", "sensor"]
    rows = []
    for rank, (score, meta) in enumerate(top_results, 1):
        rows.append({
            "rank": rank,
            "query": query_label,
            "object": query_constraints.get("object", ""),
            "location": query_constraints.get("location", ""),
            "near_feature": query_constraints.get("near_feature", ""),
            "radius_m": query_constraints.get("radius_m", ""),
            "score": round(score, 4),
            "tile_file": meta["tile_file"],
            "acquisition_date": meta["acquisition_date"],
            "lat_min": meta["lat_min"],
            "lat_max": meta["lat_max"],
            "lon_min": meta["lon_min"],
            "lon_max": meta["lon_max"],
            "sensor": meta["sensor"],
        })
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if top_results:
        safe_label = query_label.replace(" ", "_").replace("/", "-")[:40]
        out_png = os.path.join(RESULTS_DIR, f"search_{safe_label}.png")
        save_results_preview(top_results, query_label, out_png)
    print(f"Results saved -> {csv_file}")


if __name__ == "__main__":
    main()
