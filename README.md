# 🛰️ SIH PS-26227 — Satellite Intelligence System
### Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery
**Ministry of Defence (DGIS) | Smart India Hackathon 2026**

---

## 📌 Problem Statement

**PS ID**: 26227 | **Organization**: Ministry of Defence — DGIS (Indian Army)

Earth-observation archives are expanding rapidly. Conventional catalogues search only by metadata (date, coordinates). Analysts need to:
- Search satellite imagery **by meaning** — *"show me construction sites near rivers"*
- Detect **real changes** across years (not fake seasonal ones)
- Do all of this **on-premises** — no internet, no cloud, sovereign operation

---

## 🚀 Our Solution

A full end-to-end satellite intelligence pipeline — **8 phases, fully air-gapped**:

```
Download → Tile → Embed → Search → Detect Changes → Filter → Cluster → Dashboard
```

| Feature | Implementation |
|---|---|
| 🔍 Semantic text search | OpenCLIP ViT-B/32 + FAISS vector index |
| 🧭 Constraint-aware queries | Offline object, date, Ranchi/coordinate, and water-proximity parsing |
| 🖼️ Image-to-image search | CLIP visual encoder + cosine similarity |
| 🛰️ Change detection | Same-season embedding distance + measured spectral-change regions |
| 🕒 Multi-temporal evidence | First appearance and baseline-relative persistence timeline |
| 🚫 False alarm suppression | Explainable cloud/no-data, shadow, vegetation/season, and temporary-water reasons |
| 🌐 Unsupervised discovery | KMeans 8-cluster terrain grouping |
| 📋 Analyst dashboard | Streamlit review UI with offline event map and before/after evidence |
| 📑 Audit trail | JSON + CSV decision logs, full provenance |
| ⚡ Query latency | **88ms mean** (sub-100ms SLA met) |

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Satellite tiles | 180 clean 512×512 chips |
| Coverage area | 868 km² (Ranchi, Jharkhand) |
| Temporal range | 2020 — 2024 (6 epochs) |
| Changes detected | 34 raw → 9 high-confidence |
| Mean query latency | 88.28 ms |
| Terrain clusters | 8 (KMeans) |
| Total data | 347 MB |

---

## 🗂️ Folder Structure

```
SIH-PS-26227/
│
├── CODE/                               ← All Python source files
│   ├── Data_download.py                ← Phase 1: Copernicus API download
│   ├── tile_images.py                  ← Phase 2: 512×512 tiling
│   ├── build_embeddings_index.py       ← Phase 3: CLIP embeddings + FAISS
│   ├── semantic_search.py              ← Phase 4: Text/image search
│   ├── change_detection.py             ← Phase 5: Multi-temporal analysis
│   ├── false_alarm_suppression.py      ← Phase 6: Confidence calibration
│   ├── clustering_discovery.py         ← Phase 7A: KMeans clustering
│   ├── analyst_dashboard.py            ← Phase 7B: Streamlit UI
│   └── eval_report.py                  ← Phase 8: Performance evaluation
│
├── Dataset/                            ← Satellite data (Git LFS)
│   ├── Tiles/                          ← 180 × 512px GeoTIFF tiles
│   ├── Index/                          ← FAISS index + metadata JSONs
│   ├── change_previews/                ← 34 × 3-panel change evidence PNGs
│   └── search_results/                 ← Search result images + CSV
│
└── PPT_Material/                       ← Presentation assets
    ├── SIH_26227_Complete_Document.md  ← Full technical reference
    ├── 01_Change_Detection_Outputs/    ← Change evidence images
    ├── 02_Semantic_Search_Outputs/     ← Search result images
    ├── 03_System_Architecture/         ← Pipeline diagram
    ├── 04_Code_Snapshots/              ← Source code copies
    └── 05_Results_Data/                ← CSVs + evaluation report
```

---

## 🛠️ Tech Stack

| Category | Library | Version |
|---|---|---|
| Language | Python | 3.11.9 |
| AI / Embeddings | open-clip-torch (ViT-B/32) | 3.3.0 |
| Deep Learning | PyTorch | 2.14.0 |
| Vector Search | faiss-cpu | 1.15.0 |
| Geospatial I/O | rasterio | 1.4.4 |
| Coordinate System | pyproj | 3.7.2 |
| Scientific Computing | numpy, scipy | 2.4.6, 1.17.1 |
| ML Clustering | scikit-learn | 1.9.0 |
| Image Processing | Pillow | 12.3.0 |
| Dashboard / UI | Streamlit | 1.63.0 |
| Data API | openeo (Copernicus) | 0.52.0 |
| Data Catalogue | pystac | 1.15.2 |

---

## ⚙️ How to Run

### Prerequisites
- Python 3.11
- Git with Git LFS (`git lfs install`)

### Setup

```bash
# 1. Clone repo (LFS files download automatically)
git clone https://github.com/MR-ROGUE01/SIH-PS-26227.git
cd SIH-PS-26227

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install torch torchvision open-clip-torch faiss-cpu rasterio pyproj numpy pandas pillow streamlit scikit-learn scipy openeo pystac requests tqdm
```

### Run Pipeline (in order)

```bash
# Phase 1 — Download satellite data (needs Copernicus account)
python CODE\Data_download.py

# Phase 2 — Tile images into 512×512 chips
python CODE\tile_images.py

# Phase 3 — Build CLIP embeddings + FAISS index
python CODE\build_embeddings_index.py

# Phase 4 — Semantic search
python CODE\semantic_search.py --query "road development area" --top 5
python CODE\semantic_search.py --query "new construction within 500m of a river in Ranchi from 2021 to 2024" --top 5

# Phase 5 — Change detection
python CODE\change_detection.py

# Phase 6 — False alarm suppression
python CODE\false_alarm_suppression.py --final_threshold 0.30

# Phase 7A — Clustering
python CODE\clustering_discovery.py --build_clusters

# Phase 7B — Launch analyst dashboard
streamlit run CODE\analyst_dashboard.py
# Opens at http://localhost:8501
```

> **Note**: If you cloned with Git LFS, `Dataset/` is already populated. Skip Phases 1–3.

---

## 📸 Sample Outputs

### Change Detection (3-Panel Evidence)
> BEFORE | AFTER (red bounding boxes on changed objects) | CHANGE HEATMAP

Each connected changed region includes pixel area, estimated square-meter area,
centroid, local confidence, and distance to spectrally detected water where present.
The review queue includes later observations compared with the pre-change baseline;
these image-derived measurements are evidence for analyst review, not ground truth.

See `Dataset/change_previews/` and `PPT_Material/01_Change_Detection_Outputs/`

### Semantic Search Results
> Text query → Top-5 matching satellite tiles with similarity scores

See `Dataset/search_results/` and `PPT_Material/02_Semantic_Search_Outputs/`

---

## 🗺️ Area of Interest

| Parameter | Value |
|---|---|
| Location | Ranchi, Jharkhand, India |
| Bounding Box | 85.15°E–85.45°E, 23.20°N–23.45°N |
| Coverage | ~868 km² |
| Satellite | Sentinel-2 MSI L2A (10m resolution) |
| Bands | B02 Blue, B03 Green, B04 Red, B08 NIR |

---

## 🎯 Innovation Highlights

1. **Same-Season Comparison** — eliminates seasonal false alarms (crop cycles, vegetation)
2. **Zero-shot Foundation Model** — OpenCLIP used without fine-tuning on satellite data
3. **Dual-modal Search** — text AND image queries on the same FAISS index
4. **Auto Object Marking** — scipy morphology draws red bounding boxes on changed objects
5. **Fully Air-Gapped** — zero internet at inference, sovereign on-premises deployment
6. **Sub-100ms Latency** — 88ms mean on CPU-only hardware
7. **Full Audit Trail** — every analyst decision preserved with timestamp + provenance

---

## 📋 PS Requirement Coverage

| PS Section | Status |
|---|---|
| 2.2.1 Semantic & multimodal retrieval | ✅ |
| 2.2.2 Multi-temporal change analysis | ✅ |
| 2.2.3 False alarm suppression | ✅ |
| 2.2.4 Unsupervised discovery & clustering | ✅ |
| 2.2.5 Ranked analyst review queue | ✅ |
| 2.2.6 On-premises / air-gapped operation | ✅ |
| 2.2.7 Audit trail & provenance | ✅ |
| 2.3 Reproducible evaluation (<100ms) | ✅ |

---

*SIH 2026 | PS-26227 | Ministry of Defence (DGIS)*
