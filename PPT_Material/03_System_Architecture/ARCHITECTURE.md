# System Architecture — SIH PS-26227

```
┌─────────────────────────────────────────────────────────────────────┐
│              SATELLITE INTELLIGENCE SYSTEM — SIH 26227              │
│         Ministry of Defence (DGIS) | Ranchi, Jharkhand AOI          │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐
  │ PHASE 1      │   Sentinel-2 L2A GeoTIFF download (Copernicus API)
  │ Data Ingest  │   6 temporal epochs: Jan2020, Jun2020, Jun2021,
  │              │   Dec2022, Jun2023, Dec2024  (~67MB each)
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │ PHASE 2      │   512×512 tile grid (integer division, no black padding)
  │ Tiling       │   180 clean tiles → Dataset/Tiles/
  │              │   tile_catalogue.csv (lat/lon/date per tile)
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │ PHASE 3      │   OpenCLIP (ViT-B/32) — on-premises, air-gapped
  │ Embeddings   │   180 × 512-dim vectors
  │ + FAISS      │   FAISS IndexFlatIP (cosine similarity)
  └──────┬───────┘
         │
  ┌──────▼───────┐     ┌─────────────────────────────────────┐
  │ PHASE 4      │     │  QUERIES SUPPORTED:                  │
  │ Semantic     │◄────│  • Text: "road development area"     │
  │ Search       │     │  • Image: any tile file              │
  │              │     │  • Date filter: by year              │
  └──────┬───────┘     └─────────────────────────────────────┘
         │
  ┌──────▼───────┐
  │ PHASE 5      │   Same-season-only comparison (DRY_WINTER / MONSOON)
  │ Change       │   Embedding distance + spectral delta
  │ Detection    │   Change types: construction, road_development,
  │              │   clearance, water_extent
  │              │   34 changes detected, bounding box marking
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │ PHASE 6      │   Seasonal mismatch penalty (–0.30 cross-season)
  │ False Alarm  │   Quality factor (nodata fraction masking)
  │ Suppression  │   9 high-confidence candidates → review_queue.json
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │ PHASE 7A     │   KMeans (8 clusters) on 180 tile embeddings
  │ Discovery &  │   Silhouette-scored terrain grouping
  │ Clustering   │   "Find Similar Sites" via FAISS cosine search
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │ PHASE 7B     │   Streamlit dashboard (4 tabs):
  │ Analyst      │   Tab1: Change Review Queue (Confirm/Reject)
  │ Dashboard    │   Tab2: Smart Search (text + image)
  │              │   Tab3: Find Similar Locations
  │              │   Tab4: Analyst Decisions audit trail
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │ PHASE 8      │   evaluation_report.md / .json
  │ Eval Report  │   Mean query latency: 88ms | Coverage: 868 km²
  └──────────────┘
```

## Offline Evidence Extensions

- Search parses object terms, explicit years/dates, the Ranchi AOI or coordinates, and river/water proximity locally; CLIP still ranks the original query text.
- Change candidates include connected-region area, centroid, local confidence, water distance, and a baseline-relative observation timeline.
- Suppression retains the audit trail and records cloud/no-data, possible shadow, vegetation/seasonal variation, and temporary-water cues with their penalties.
- The review map is a coordinate plot rather than a basemap, so it does not request external map tiles.
