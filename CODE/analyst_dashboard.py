"""
Satellite Intelligence Dashboard
SIH Problem Statement 26227 — Ministry of Defence

Run:
    streamlit run CODE\\analyst_dashboard.py
"""

import base64
import html
import io
import math
import os
import sys
import json
import datetime
import subprocess
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# ─── FIX: Add CODE folder to Python path so imports work ─────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from query_understanding import parse_query

# ─── PATH RESOLUTION ─────────────────────────────────────────────────────────
SIH_DIR      = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR).upper() == "CODE" else SCRIPT_DIR
DATASET_DIR  = os.path.join(SIH_DIR, "Dataset")
INDEX_DIR    = os.path.join(DATASET_DIR, "Index")
TILES_DIR    = os.path.join(DATASET_DIR, "Tiles")
PREVIEWS_DIR = os.path.join(DATASET_DIR, "change_previews")
SEARCH_DIR   = os.path.join(DATASET_DIR, "search_results")

REVIEW_QUEUE_FILE = os.path.join(INDEX_DIR, "review_queue.json")
CLUSTERS_FILE     = os.path.join(INDEX_DIR, "tile_clusters.json")
DECISIONS_FILE    = os.path.join(DATASET_DIR, "analyst_decisions.json")

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Satellite Intelligence — SIH 26227",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        --ops-bg: #070b12;
        --ops-panel: #0d1521;
        --ops-panel-raised: #121e2b;
        --ops-line: #1d3345;
        --ops-text: #e5f0f5;
        --ops-muted: #8298a8;
        --ops-cyan: #4bd5e8;
        --ops-green: #72d6a0;
        --ops-amber: #e8b65b;
        --ops-red: #ec7784;
    }
    [data-testid="stAppViewContainer"] {
        background-color: var(--ops-bg);
        background-image:
            linear-gradient(rgba(75, 213, 232, 0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(75, 213, 232, 0.025) 1px, transparent 1px),
            radial-gradient(ellipse at 70% 0%, rgba(22, 103, 131, 0.12), transparent 44%);
        background-size: 36px 36px, 36px 36px, auto;
        color: var(--ops-text);
    }
    [data-testid="stHeader"] {
        background: transparent !important;
        box-shadow: none !important;
        border: 0 !important;
    }
    [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
    .block-container { max-width: none; padding: 0.8rem 1.2rem 2.5rem; }
    [data-testid="stSidebar"] {
        background: #0a111b;
        border-right: 1px solid var(--ops-line);
        min-width: 215px;
        max-width: 250px;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
    [data-testid="stMetric"] {
        min-height: 88px;
        padding: 12px 14px;
        background: linear-gradient(145deg, rgba(15, 28, 40, 0.96), rgba(10, 18, 28, 0.96));
        border: 1px solid var(--ops-line);
        border-radius: 5px;
        box-shadow: inset 0 1px rgba(132, 209, 225, 0.04), 0 8px 22px rgba(0, 0, 0, 0.2);
    }
    [data-testid="stMetricLabel"] p {
        color: var(--ops-muted);
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    [data-testid="stMetricValue"] { color: var(--ops-cyan); font-family: Bahnschrift, "Segoe UI", sans-serif; }
    [data-baseweb="tab-list"] {
        gap: 0.2rem;
        border-bottom: 1px solid rgba(67, 116, 139, 0.38);
    }
    [data-baseweb="tab"] {
        height: 2.65rem;
        color: var(--ops-muted);
        background: transparent;
        border-radius: 3px 3px 0 0;
    }
    [data-baseweb="tab"][aria-selected="true"] {
        color: var(--ops-cyan);
        background: rgba(18, 40, 53, 0.55);
    }
    [data-testid="stRadio"] > div { gap: 0.4rem; }
    [data-testid="stRadio"] label {
        padding: 0.35rem 0.62rem;
        background: rgba(10, 23, 35, 0.76);
        border: 1px solid #1d394b;
        border-radius: 4px;
    }
    [data-testid="stRadio"] label:has(input:checked) {
        border-color: rgba(75, 213, 232, 0.6);
        box-shadow: 0 0 12px rgba(75, 213, 232, 0.1);
    }
    [data-testid="stButton"] button {
        min-height: 2.35rem;
        border-radius: 4px;
        border: 1px solid var(--ops-line);
        transition: border-color 120ms ease, background-color 120ms ease;
    }
    [data-testid="stButton"] button:hover { border-color: var(--ops-cyan); box-shadow: 0 0 12px rgba(75, 213, 232, 0.12); }
    [data-testid="stButton"] button[kind="primary"] {
        color: #041117;
        background: var(--ops-cyan);
        border-color: var(--ops-cyan);
        font-weight: 700;
    }
    [data-testid="stExpander"] {
        background: rgba(10, 20, 31, 0.82);
        border: 1px solid var(--ops-line);
        border-radius: 4px;
    }
    [data-testid="stDataFrame"] { border: 1px solid var(--ops-line); border-radius: 4px; }
    [data-testid="stImage"] img { border: 1px solid #244455; border-radius: 3px; }
    [data-testid="stTextInput"] input, [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: #09131e;
        border-color: #244052;
        border-radius: 4px;
    }
    [data-testid="stTextInput"] input { min-height: 2.75rem; color: var(--ops-text); }
    [data-testid="stTextInput"] input:focus { border-color: var(--ops-cyan); box-shadow: 0 0 0 1px rgba(75, 213, 232, .36), 0 0 18px rgba(75, 213, 232, .1); }
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background: linear-gradient(145deg, rgba(12, 24, 36, .9), rgba(7, 15, 24, .95));
        border-color: rgba(75, 213, 232, .2);
        border-radius: 5px;
    }
    [data-testid="stSlider"] [role="slider"] { background: var(--ops-cyan); }
    .map-side-readout { min-height: 610px; padding: .9rem; background: linear-gradient(155deg, rgba(10, 23, 35, .92), rgba(7, 14, 23, .96)); border: 1px solid rgba(75, 213, 232, .2); border-radius: 5px; box-shadow: inset 0 1px rgba(151, 229, 237, .05), 0 0 24px rgba(36, 142, 166, .06); }
    .map-side-label { margin-bottom: .3rem; color: #7894a4; font-size: .62rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
    .map-side-value { margin-bottom: .9rem; color: #e5f0f5; font-family: Bahnschrift, "Segoe UI", sans-serif; font-size: .92rem; font-weight: 700; overflow-wrap: anywhere; }
    .map-side-rule { height: 1px; margin: .7rem 0; background: linear-gradient(90deg, rgba(75, 213, 232, .35), rgba(29, 51, 69, .4)); }
    .map-heading {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.6rem;
        color: var(--ops-muted);
        font-family: "Cascadia Code", Consolas, monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
    }
    .map-heading strong { color: var(--ops-text); font-family: Bahnschrift, "Segoe UI", sans-serif; font-size: .95rem; }
    .glass-panel {
        padding: 1rem;
        background: linear-gradient(145deg, rgba(12, 24, 36, 0.9), rgba(7, 15, 24, 0.95));
        border: 1px solid rgba(75, 213, 232, 0.2);
        border-radius: 5px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.22);
    }
    .control-kicker {
        margin: 0 0 0.6rem;
        color: var(--ops-cyan);
        font-family: "Cascadia Code", Consolas, monospace;
        font-size: 0.67rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .map-status {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        color: var(--ops-green);
        font-family: "Cascadia Code", Consolas, monospace;
        font-size: 0.66rem;
    }
    .map-status span { width: 6px; height: 6px; background: var(--ops-green); border-radius: 50%; box-shadow: 0 0 8px rgba(114, 214, 160, 0.75); }
    .metric-strip { margin: 0.5rem 0 1rem; }
    .evidence-heading {
        margin: 1.1rem 0 0.55rem;
        color: var(--ops-muted);
        font-family: "Cascadia Code", Consolas, monospace;
        font-size: 0.69rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .mission-banner {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
        margin: 0 0 1.5rem;
        padding: 1.25rem 1.4rem;
        background: linear-gradient(105deg, rgba(10, 23, 35, 0.98), rgba(8, 14, 23, 0.96));
        border: 1px solid rgba(75, 213, 232, 0.19);
        border-left: 3px solid var(--ops-cyan);
        border-radius: 4px;
        box-shadow: inset 0 1px rgba(151, 229, 237, 0.04), 0 8px 24px rgba(0, 0, 0, 0.22);
    }
    .mission-kicker {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin-bottom: 0.35rem;
        color: var(--ops-cyan);
        font-size: 0.69rem;
        font-weight: 800;
        text-transform: uppercase;
    }
    .mission-dot { width: 6px; height: 6px; background: var(--ops-green); border-radius: 50%; box-shadow: 0 0 8px rgba(114, 214, 160, 0.8); }
    .mission-title { color: var(--ops-text); font-family: Bahnschrift, "Segoe UI", sans-serif; font-size: 1.25rem; font-weight: 700; }
    .mission-subtitle { margin-top: 0.1rem; color: var(--ops-muted); font-size: 0.76rem; }
    .mission-readout { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 0.5rem; }
    .readout-chip {
        padding: 0.48rem 0.65rem;
        color: #c4d8e3;
        background: rgba(4, 12, 20, 0.62);
        border: 1px solid #203c4e;
        border-radius: 3px;
        font-family: "Cascadia Code", Consolas, monospace;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        white-space: nowrap;
    }
    .readout-chip.accent { color: var(--ops-green); border-color: rgba(114, 214, 160, .34); }
    .readout-chip.online { display: inline-flex; align-items: center; gap: .4rem; color: var(--ops-green); border-color: rgba(114, 214, 160, .28); }
    .readout-chip.online .mission-dot { display: inline-block; flex: 0 0 6px; margin: 0; }
    .search-kicker { margin: 0 0 .6rem; color: var(--ops-cyan); font-family: "Cascadia Code", Consolas, monospace; font-size: .68rem; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; }
    .mod-badge {
        background: #102332;
        color: var(--ops-cyan);
        padding: 5px 9px;
        border: 1px solid #275065;
        border-radius: 3px;
        font-weight: 700;
        font-size: 0.68rem;
        display: inline-block;
        margin-bottom: 8px;
        letter-spacing: 0.04em;
    }
    @media (max-width: 760px) {
        .mission-banner { align-items: flex-start; flex-direction: column; padding: 1rem; }
        .mission-readout { justify-content: flex-start; }
        .mission-title { font-size: 1.1rem; }
        .block-container { padding: 0.55rem 0.55rem 2rem; }
        .map-side-readout { min-height: 0; }
    }
</style>
""", unsafe_allow_html=True)


# ─── DECISIONS FILE INIT ─────────────────────────────────────────────────────

if not os.path.exists(DECISIONS_FILE):
    with open(DECISIONS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2)

def load_decisions():
    try:
        with open(DECISIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_decision(candidate_id, decision, notes, candidate_data):
    decisions = load_decisions()
    entry = {
        "candidate_id": candidate_id,
        "decision": decision,
        "analyst_notes": notes,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "change_type": candidate_data.get("change_type"),
        "confidence": candidate_data.get("adjusted_confidence", candidate_data.get("confidence")),
        "before_date": candidate_data.get("before_date"),
        "after_date": candidate_data.get("after_date"),
        "before_tile": candidate_data.get("before_tile", candidate_data.get("before_file")),
        "after_tile": candidate_data.get("after_tile", candidate_data.get("after_file")),
        "lat": candidate_data.get("lat_min"),
        "lon": candidate_data.get("lon_min"),
        "sensor": candidate_data.get("sensor", "Sentinel-2 L2A")
    }
    existing_ids = [d.get("candidate_id") for d in decisions]
    if candidate_id in existing_ids:
        for idx, d in enumerate(decisions):
            if d.get("candidate_id") == candidate_id:
                decisions[idx] = entry
    else:
        decisions.append(entry)
    with open(DECISIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2)


def prepare_candidate_map(candidate):
    preview_file = candidate.get("preview_file")
    preview_path = os.path.join(PREVIEWS_DIR, preview_file) if preview_file else None
    if not preview_path or not os.path.exists(preview_path):
        return None

    try:
        with Image.open(preview_path) as source:
            preview = source.convert("RGB")
        panel_width = preview.width // 3
        imagery_height = max(1, preview.height - 65)
        after_image = preview.crop((panel_width, 0, panel_width * 2, imagery_height))
        buffer = io.BytesIO()
        after_image.save(buffer, format="PNG")
        image_uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

        bounds = [
            float(candidate["lon_min"]), float(candidate["lat_min"]),
            float(candidate["lon_max"]), float(candidate["lat_max"]),
        ]
        center_lon = (bounds[0] + bounds[2]) / 2
        center_lat = (bounds[1] + bounds[3]) / 2
        regions = candidate.get("objects") or [None]
        markers = []
        for index, region in enumerate(regions, 1):
            area_m2 = float(region.get("area_m2", 0)) if region else 0.0
            markers.append({
                "lon": float(region["centroid_lon"]) if region else center_lon,
                "lat": float(region["centroid_lat"]) if region else center_lat,
                "object_id": region.get("object_id", index) if region else "Candidate center",
                "confidence": float(region.get("confidence", candidate.get("adjusted_confidence", candidate.get("confidence", 0)))) if region else float(candidate.get("adjusted_confidence", candidate.get("confidence", 0))),
                "area_m2": area_m2,
                "radius_m": max(45, min(220, (area_m2 ** 0.5) * 0.35)) if area_m2 else 90,
            })
        return image_uri, bounds, markers, preview
    except (OSError, KeyError, TypeError, ValueError):
        return None


def candidate_map_html(candidate, map_data):
    image_uri, bounds, markers, _ = map_data
    west, south, east, north = bounds
    map_width_m = (east - west) * 111_320 * math.cos(math.radians((south + north) / 2))
    change_type = html.escape(candidate.get("change_type", "CHANGE").replace("_", " ").upper())
    after_date = html.escape(candidate.get("after_date", ""))
    marker_html = []
    for marker in markers:
        left = max(0, min(100, (marker["lon"] - west) / (east - west) * 100))
        top = max(0, min(100, (north - marker["lat"]) / (north - south) * 100))
        label = html.escape(
            f"Region {marker['object_id']} · Confidence {marker['confidence']:.2f} · {marker['area_m2']:.0f} m²"
        )
        marker_html.append(
            f'<button class="map-marker" style="left:{left:.4f}%;top:{top:.4f}%" '
            f'title="{label}" aria-label="{label}"></button>'
        )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<style>
*{{box-sizing:border-box}}html,body{{margin:0;width:100%;height:100%;background:#111813;color:#e8eee9}}
.local-map-frame{{position:relative;width:100%;height:610px;overflow:hidden;background:#070d15;border:1px solid #1e5167;border-radius:4px;font:10px 'Cascadia Code',Consolas,monospace;box-shadow:inset 0 0 42px rgba(23,135,162,.08),0 0 22px rgba(25,144,175,.08)}}
.map-viewport{{position:absolute;inset:0;overflow:hidden;touch-action:none;cursor:grab}}.map-viewport.dragging{{cursor:grabbing}}
.map-world{{position:absolute;inset:0;transform-origin:center;will-change:transform}}
.map-world>img{{display:block;width:100%;height:100%;object-fit:fill;user-select:none;pointer-events:none}}
.map-grid{{position:absolute;inset:0;pointer-events:none;opacity:.3;background-image:linear-gradient(rgba(90,199,221,.17) 1px,transparent 1px),linear-gradient(90deg,rgba(90,199,221,.17) 1px,transparent 1px);background-size:20% 20%}}
.map-vignette{{position:absolute;inset:0;pointer-events:none;background:radial-gradient(ellipse at center,transparent 42%,rgba(2,7,13,.52) 100%)}}
.map-marker{{position:absolute;display:grid;width:20px;height:20px;padding:0;place-items:center;transform:translate(-50%,-50%);background:rgba(75,213,232,.18);border:1px solid rgba(111,235,244,.98);border-radius:50%;box-shadow:0 0 0 5px rgba(75,213,232,.12),0 0 19px rgba(75,213,232,.8);cursor:help;animation:marker-pulse 2.4s ease-out infinite}}
@keyframes marker-pulse{{0%,100%{{box-shadow:0 0 0 4px rgba(75,213,232,.08),0 0 12px rgba(75,213,232,.48)}}50%{{box-shadow:0 0 0 7px rgba(75,213,232,.16),0 0 22px rgba(75,213,232,.85)}}}}
.map-marker:after{{content:'';width:5px;height:5px;background:#baf7ff;border-radius:50%}}
.hud{{position:absolute;z-index:2;color:#d5eaf0;text-shadow:0 1px 5px #03080d;pointer-events:none}}
.tl{{top:14px;left:15px;display:grid;gap:4px}}.tr{{top:14px;right:15px;text-align:right}}.bl{{bottom:14px;left:15px;display:grid;gap:3px}}
.tag{{color:#55d9ec;font-weight:800;letter-spacing:.08em}}.coords{{color:#c3dce4}}
.controls{{position:absolute;z-index:3;right:14px;bottom:14px;display:grid;gap:5px;pointer-events:auto}}
.controls button{{width:34px;height:34px;color:#dff8fc;background:rgba(5,14,22,.92);border:1px solid #286076;border-radius:3px;font-size:17px;cursor:pointer}}
.controls button:hover{{color:#59dced;border-color:#55d9ec;box-shadow:0 0 11px rgba(75,213,232,.22)}}.north{{position:absolute;z-index:2;top:48px;right:20px;color:#8bdce9;text-align:center;text-shadow:0 1px 4px #03080d}}
.north b{{display:block;font-size:20px;line-height:18px}}.scale{{display:inline-flex;align-items:center;gap:5px}}
.scale i{{display:block;width:36px;height:5px;border:1px solid #e8eee9;border-top:0}}
</style></head><body>
 <div class="local-map-frame" id="map" data-width-meters="{map_width_m:.1f}">
 <div class="map-viewport" id="viewport"><div class="map-world" id="world">
  <img src="{image_uri}" alt="Selected candidate after image with detected change regions">
  <div class="map-grid"></div>{''.join(marker_html)}<div class="map-vignette"></div>
 </div></div>
 <div class="hud tl"><span class="tag">LIVE SCENE OVERLAY</span><span>RANCHI · JHARKHAND</span></div>
 <div class="hud tr"><span class="tag">{change_type}</span><br>{after_date}</div>
 <div class="north">N<b>↑</b></div>
 <div class="hud bl"><span class="coords">{north:.4f}° N &nbsp; / &nbsp; {west:.4f}° E</span><span class="scale"><i id="scale-bar"></i><span id="scale-label"></span></span></div>
 <div class="controls"><button id="zoom-in" title="Zoom in" aria-label="Zoom in">+</button><button id="zoom-out" title="Zoom out" aria-label="Zoom out">−</button><button id="reset" title="Reset view" aria-label="Reset view">⌂</button></div>
</div>
<script>
const viewport=document.getElementById('viewport'),world=document.getElementById('world');let zoom=1,x=0,y=0,drag=null;
const mapWidthMeters=Number(document.getElementById('map').dataset.widthMeters);
function updateScale(){{const target=mapWidthMeters/zoom*.16,unit=10**Math.floor(Math.log10(target)),distance=[5,2,1].map(value=>value*unit).find(value=>value<=target)||unit;document.getElementById('scale-bar').style.width=`${{viewport.clientWidth*distance*zoom/mapWidthMeters}}px`;document.getElementById('scale-label').textContent=distance>=1000?`${{(distance/1000).toFixed(distance%1000?1:0)}} km`:`${{Math.round(distance)}} m`}}
function update(){{world.style.transform=`translate(${{x}}px,${{y}}px) scale(${{zoom}})`;updateScale()}}
document.getElementById('zoom-in').onclick=()=>{{zoom=Math.min(4,zoom*1.25);update()}};
document.getElementById('zoom-out').onclick=()=>{{zoom=Math.max(1,zoom/1.25);if(zoom===1){{x=0;y=0}}update()}};
document.getElementById('reset').onclick=()=>{{zoom=1;x=0;y=0;update()}};
viewport.addEventListener('wheel',event=>{{event.preventDefault();zoom=Math.max(1,Math.min(4,zoom*(event.deltaY<0?1.12:0.89)));update()}},{{passive:false}});
viewport.addEventListener('pointerdown',event=>{{if(event.target.closest('button'))return;drag={{x:event.clientX-x,y:event.clientY-y}};viewport.classList.add('dragging');viewport.setPointerCapture(event.pointerId)}});
viewport.addEventListener('pointermove',event=>{{if(!drag)return;x=event.clientX-drag.x;y=event.clientY-drag.y;update()}});
viewport.addEventListener('pointerup',()=>{{drag=null;viewport.classList.remove('dragging')}});
window.addEventListener('resize',updateScale);updateScale();
</script></body></html>"""


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="mod-badge">MINISTRY OF DEFENCE | DGIS</div>', unsafe_allow_html=True)
    st.title("🛰️ SIH PS-26227")
    st.caption("Satellite Semantic Retrieval & Change Analysis")
    st.divider()

    st.markdown("**Area Details:**")
    st.markdown("- **Location**: Ranchi, Jharkhand")
    st.markdown("- **Total Tiles**: 180 (512×512 each)")
    st.markdown("- **Years Covered**: 2020 — 2024")
    st.markdown("- **Satellite**: Sentinel-2 L2A (10m)")
    st.divider()

    decisions = load_decisions()
    confirmed_count = sum(1 for d in decisions if d["decision"] == "CONFIRMED")
    rejected_count  = sum(1 for d in decisions if d["decision"] == "REJECTED")
    st.metric("✅ Confirmed Changes", confirmed_count)
    st.metric("❌ Rejected (False Alarms)", rejected_count)


st.markdown(f"""
<section class="mission-banner">
    <div>
        <div class="mission-kicker"><span class="mission-dot"></span> DGIS / EARTH OBSERVATION</div>
        <div class="mission-title">Ranchi Change Intelligence</div>
        <div class="mission-subtitle">Ranchi, Jharkhand &nbsp;·&nbsp; Multi-temporal archive review</div>
    </div>
    <div class="mission-readout">
        <span class="readout-chip online"><span class="mission-dot"></span> SYSTEM ONLINE</span>
        <span class="readout-chip accent">AIR-GAPPED</span>
        <span class="readout-chip">SENTINEL-2 / L2A</span>
        <span class="readout-chip">2020—2024 &nbsp;·&nbsp; 6 EPOCHS</span>
        <span class="readout-chip">CONFIRMED &nbsp;{confirmed_count}</span>
        <span class="readout-chip">FALSE ALARMS &nbsp;{rejected_count}</span>
    </div>
</section>
""", unsafe_allow_html=True)


# ─── MAIN TABS ───────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Change Review Queue",
    "🔍 Smart Search",
    "🌐 Find Similar Locations",
    "📑 Analyst Decisions"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: CHANGE REVIEW QUEUE
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("📋 Change Review Queue")
    st.caption("Review and verify multi-temporal change detections — Validate or dismiss flagged anomalies")

    if not os.path.exists(REVIEW_QUEUE_FILE):
        st.warning("Review queue not found. Please execute the suppression pipeline: `python CODE/false_alarm_suppression.py`")
    else:
        with open(REVIEW_QUEUE_FILE, "r", encoding="utf-8") as f:
            queue = json.load(f)

        if not queue:
            st.info("No candidates currently pending review.")
        else:
            controls_col, map_col = st.columns([0.82, 4.18], gap="medium")
            with controls_col:
                with st.container(border=True):
                    st.markdown('<div class="control-kicker">Review controls</div>', unsafe_allow_html=True)
                    queue_search = st.text_input(
                        "Find an event",
                        placeholder="Type, date, tile or coordinate",
                        label_visibility="collapsed",
                    )
                    type_options = ["ALL"] + sorted(list(set(c["change_type"] for c in queue)))
                    selected_type = st.selectbox("Change type", type_options)
                    min_conf = st.slider("Minimum confidence", 0.20, 0.95, 0.35, 0.05)

                query_text = queue_search.strip().lower()
                filtered_queue = [
                    candidate for candidate in queue
                    if (selected_type == "ALL" or candidate["change_type"] == selected_type)
                    and candidate.get("adjusted_confidence", candidate.get("confidence", 0)) >= min_conf
                    and (not query_text or query_text in " ".join(str(candidate.get(key, "")) for key in (
                        "change_type", "before_date", "after_date", "before_tile", "after_tile",
                        "row", "col", "lat_min", "lon_min",
                    )).lower())
                ]
                st.metric("Events in view", len(filtered_queue))

                options = [
                    f"{c['change_type'].replace('_', ' ').upper()} · {c['before_date']} → {c['after_date']} · {c.get('adjusted_confidence', c['confidence']):.2f}"
                    for c in filtered_queue
                ]
                if options:
                    selected_idx = st.selectbox(
                        "Selected event",
                        range(len(options)),
                        format_func=lambda index: options[index],
                    )
                    cand = filtered_queue[selected_idx]
                    cand_id = f"{cand.get('row', 0)}_{cand.get('col', 0)}_{cand['before_date']}_{cand['after_date']}"
                    st.markdown(
                        f"<div class='control-kicker'>TARGET · {cand.get('row', 0):02d}/{cand.get('col', 0):02d}</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"{cand.get('sensor', 'Sentinel-2 L2A')}  /  {cand.get('season', 'ARCHIVE')}")
                else:
                    cand = None

            with map_col:
                if cand:
                    map_data = prepare_candidate_map(cand)
                    st.markdown(
                        f"<div class='map-heading'><span><strong>SCENE OVERLAY</strong> &nbsp; / &nbsp; "
                        f"{cand['before_date']} → {cand['after_date']}</span>"
                        f"<span class='map-status'><span></span> LOCAL IMAGERY · EPSG:4326</span></div>",
                        unsafe_allow_html=True,
                    )
                    if map_data:
                        _, bounds, markers, preview = map_data
                        map_view_col, readout_col = st.columns([3.0, 0.78], gap="small")
                        with map_view_col:
                            components.html(
                                candidate_map_html(cand, map_data),
                                height=610,
                                scrolling=False,
                            )
                        with readout_col:
                            lat_center = (bounds[1] + bounds[3]) / 2
                            lon_center = (bounds[0] + bounds[2]) / 2
                            st.markdown(
                                f"<div class='map-side-readout'>"
                                f"<div class='map-side-label'>Target classification</div>"
                                f"<div class='map-side-value'>{html.escape(cand['change_type'].replace('_', ' ').upper())}</div>"
                                f"<div class='map-side-label'>Observation window</div>"
                                f"<div class='map-side-value'>{html.escape(cand['before_date'])}<br>↓<br>{html.escape(cand['after_date'])}</div>"
                                f"<div class='map-side-rule'></div>"
                                f"<div class='map-side-label'>Scene center</div>"
                                f"<div class='map-side-value'>{lat_center:.4f}° N<br>{lon_center:.4f}° E</div>"
                                f"<div class='map-side-label'>Sensor / season</div>"
                                f"<div class='map-side-value'>{html.escape(cand.get('sensor', 'Sentinel-2 L2A'))}<br>{html.escape(cand.get('season', 'ARCHIVE'))}</div>"
                                f"<div class='map-side-rule'></div>"
                                f"<div class='map-side-label'>Detected regions</div>"
                                f"<div class='map-side-value'>{len(cand.get('objects') or []) or cand.get('num_objects_marked', 0)}</div>"
                                f"<div class='map-side-label'>Confidence</div>"
                                f"<div class='map-side-value'>{float(cand.get('adjusted_confidence', cand.get('confidence', 0))):.3f}</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        preview = None
                        st.warning("Scene evidence is unavailable for this event.")
                else:
                    preview = None
                    st.info("No events match these filters.")

            if filtered_queue:
                graph_rows = []
                for candidate in filtered_queue:
                    graph_rows.append({
                        "longitude": (float(candidate["lon_min"]) + float(candidate["lon_max"])) / 2,
                        "latitude": (float(candidate["lat_min"]) + float(candidate["lat_max"])) / 2,
                        "change_type": candidate["change_type"].replace("_", " ").upper(),
                        "confidence": float(candidate.get("adjusted_confidence", candidate.get("confidence", 0))),
                        "observation_date": candidate["after_date"],
                        "tile": candidate.get("after_tile", candidate.get("after_file", "")),
                    })
                st.markdown('<div class="evidence-heading">Review queue visualization · geographic distribution</div>', unsafe_allow_html=True)
                st.caption("Each point is a filtered review candidate; color shows change class and size shows confidence.")
                st.scatter_chart(
                    pd.DataFrame(graph_rows),
                    x="longitude",
                    y="latitude",
                    color="change_type",
                    size="confidence",
                    height=280,
                    width="stretch",
                )

            if cand:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Change class", cand["change_type"].replace("_", " ").upper())
                m2.metric("Confidence", f"{cand.get('adjusted_confidence', cand['confidence']):.3f}")
                area_m2 = cand.get("area_m2")
                m3.metric("Mapped area", f"{float(area_m2) / 10_000:.2f} ha" if area_m2 is not None else "N/A")
                m4.metric("First observed", cand.get("first_appeared", cand.get("earliest_observation", cand["after_date"])))

                if preview is None:
                    preview_file = cand.get("preview_file")
                    preview_path = os.path.join(PREVIEWS_DIR, preview_file) if preview_file else None
                    if preview_path and os.path.exists(preview_path):
                        preview = Image.open(preview_path).convert("RGB")
                if preview:
                    st.markdown('<div class="evidence-heading">Temporal evidence · before / after / change signature</div>', unsafe_allow_html=True)
                    panel_width = preview.width // 3
                    imagery_height = max(1, preview.height - 65)
                    image_cols = st.columns(3)
                    image_captions = ["BEFORE · " + cand["before_date"], "AFTER · " + cand["after_date"], "SPECTRAL DELTA"]
                    for panel_index, image_col in enumerate(image_cols):
                        panel = preview.crop((panel_index * panel_width, 0,
                                               (panel_index + 1) * panel_width, imagery_height))
                        image_col.image(panel, use_container_width=True, caption=image_captions[panel_index])
                else:
                    st.warning("Preview imagery unavailable.")

                with st.expander("🔍 Full Details (Location, Sensor, Suppression info)", expanded=False):
                    p_col1, p_col2 = st.columns(2)
                    with p_col1:
                        st.markdown(f"**Before Tile**: `{cand.get('before_tile', cand.get('before_file'))}`")
                        st.markdown(f"**After Tile**: `{cand.get('after_tile', cand.get('after_file'))}`")
                        st.markdown(f"**Satellite**: `{cand.get('sensor', 'Sentinel-2 L2A')}`")
                        st.markdown(f"**Embedding Distance**: `{cand.get('embedding_distance', 'N/A')}`")
                        st.markdown(f"**Date Range**: `{cand.get('before_date')} to {cand.get('after_date')}`")
                        st.markdown(f"**Persisted Through**: `{cand.get('persisted_through', 'unknown')}`")
                    with p_col2:
                        st.markdown(f"**Lat Range**: `[{cand.get('lat_min',0):.4f}, {cand.get('lat_max',0):.4f}]`")
                        st.markdown(f"**Lon Range**: `[{cand.get('lon_min',0):.4f}, {cand.get('lon_max',0):.4f}]`")
                        st.markdown(f"**Seasonal Penalty**: `{cand.get('seasonal_penalty', 0.0)}`")
                        st.markdown(f"**Suppression Flags**: `{', '.join(cand.get('suppression_reasons', ['None']))}`")
                        st.markdown(f"**Evidence Cues**: `{cand.get('false_alarm_evidence', {})}`")

                if cand.get("temporal_timeline"):
                    st.markdown("**Multi-temporal persistence**")
                    st.dataframe(pd.DataFrame(cand["temporal_timeline"]), use_container_width=True, hide_index=True)
                if cand.get("objects"):
                    st.markdown("**Detected change regions**")
                    object_columns = ["object_id", "area_m2", "confidence", "centroid_lat",
                                      "centroid_lon", "distance_to_water_m", "bbox_pixels"]
                    st.dataframe(
                        pd.DataFrame(cand["objects"]).reindex(columns=object_columns),
                        use_container_width=True, hide_index=True,
                    )

                st.subheader("Analyst Verification")
                notes_input = st.text_area("Notes (optional):", placeholder="e.g., New road confirmed near river bend.", key=f"note_{cand_id}")

                col_btn1, col_btn2, _ = st.columns([1, 1, 3])
                with col_btn1:
                    if st.button("✅ Confirm Change", type="primary", use_container_width=True):
                        save_decision(cand_id, "CONFIRMED", notes_input, cand)
                        st.success("Confirmed! Event recorded in audit trail.")
                        st.rerun()
                with col_btn2:
                    if st.button("❌ Reject (False Alarm)", use_container_width=True):
                        save_decision(cand_id, "REJECTED", notes_input, cand)
                        st.warning("Rejected! False alarm logged in audit trail.")
                        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: SMART SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("🛰️ Intelligence Search")
    st.caption("ARCHIVE TASKING  /  SEMANTIC AND REFERENCE-IMAGE RETRIEVAL")

    query_type = st.radio("Search Modality:", ["Text Query (Natural Language)", "Image Query (Reference Tile)"], horizontal=True)

    search_query = ""
    image_tile_name = ""

    if query_type == "Text Query (Natural Language)":
        st.markdown('<div class="search-kicker">Mission query · suggested targets</div>', unsafe_allow_html=True)
        q_cols = st.columns(4)
        if q_cols[0].button("🌊 Structures near river"):
            st.session_state["sq"] = "newly built structures near a river"
        if q_cols[1].button("🚜 Vehicle concentrations"):
            st.session_state["sq"] = "large vehicle concentrations on open ground"
        if q_cols[2].button("🛣️ Road development"):
            st.session_state["sq"] = "road development area"
        if q_cols[3].button("🏗️ Construction site"):
            st.session_state["sq"] = "construction site"

        with st.container(border=True):
            st.markdown('<div class="search-kicker">Natural-language command</div>', unsafe_allow_html=True)
            search_query = st.text_input(
                "Enter search prompt:",
                value=st.session_state.get("sq", "dense urban area"),
                placeholder="e.g. new construction within 500m of a river in Ranchi from 2021 to 2024",
            )
            if search_query.strip():
                parsed_query = parse_query(search_query)
                query_details = [f"Object: {parsed_query['object'] or 'semantic match'}"]
                if parsed_query["location"]:
                    query_details.append(f"Location: {parsed_query['location']}")
                if parsed_query["date_from"] or parsed_query["date_to"]:
                    query_details.append(f"Dates: {parsed_query['date_from'] or 'any'} to {parsed_query['date_to'] or 'any'}")
                if parsed_query["near_feature"]:
                    query_details.append(f"Near {parsed_query['near_feature']}: {parsed_query['radius_m']:.0f} m")
                st.caption("PARSED CONSTRAINTS  /  " + "  ·  ".join(query_details))
    else:
        all_tiles = sorted(os.listdir(TILES_DIR)) if os.path.exists(TILES_DIR) else []
        image_tile_name = st.selectbox("Select reference tile:", all_tiles)

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        top_k = st.slider("Top Results (K):", 3, 10, 5)
    with col_opt2:
        date_filter = st.selectbox("Temporal Filter (Year):", ["All Dates", "2020", "2021", "2022", "2023", "2024"])

    if st.button("⌕  SEARCH ARCHIVE", type="primary"):
        with st.spinner("Searching satellite tiles..."):
            command = [sys.executable, os.path.join(SCRIPT_DIR, "semantic_search.py")]
            if query_type == "Text Query (Natural Language)":
                command.extend(["--query", search_query])
            else:
                command.extend(["--image", image_tile_name])
            command.extend(["--top", str(top_k)])
            if date_filter != "All Dates":
                command.extend(["--date-from", f"{date_filter}-01-01", "--date-to", f"{date_filter}-12-31"])
            search_process = subprocess.run(command, cwd=SIH_DIR, capture_output=True, text=True)
            if search_process.returncode:
                st.error(search_process.stderr.strip() or search_process.stdout.strip())

        csv_path = os.path.join(SEARCH_DIR, "search_results.csv")
        if search_process.returncode == 0 and os.path.exists(csv_path):
            df_results = pd.read_csv(csv_path)
            st.success(f"{len(df_results)} matching tiles retrieved successfully.")

            cols = st.columns(min(len(df_results), 5))
            for i, (_, row) in enumerate(df_results.iterrows()):
                if i < len(cols):
                    with cols[i]:
                        st.markdown(f"**Rank #{row['rank']}**")
                        st.caption(f"Score: **{row['score']:.4f}**")
                        st.caption(f"Date: `{row['acquisition_date']}`")
                        t_path = os.path.join(TILES_DIR, row["tile_file"])
                        if os.path.exists(t_path):
                            try:
                                from semantic_search import tile_to_rgb
                                img = tile_to_rgb(t_path)
                                st.image(img, use_container_width=True, caption=row["tile_file"])
                            except Exception:
                                st.code(row["tile_file"])
                        else:
                            st.code(row["tile_file"])

            st.dataframe(df_results, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: FIND SIMILAR LOCATIONS
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("🌐 Find Similar Locations")
    st.caption("Unsupervised site discovery — retrieve locations with equivalent semantic and spectral signatures")

    subtab1, subtab2 = st.tabs(["🎯 Similar Sites Retrieval", "📊 Latent Terrain Clusters"])

    with subtab1:
        st.subheader("Reference Tile Selection")

        all_tiles = sorted(os.listdir(TILES_DIR)) if os.path.exists(TILES_DIR) else []
        chosen_tile = st.selectbox("Select Target Tile:", all_tiles, key="disc_tile")

        if st.button("⚡ Discover Similar Sites", type="primary"):
            with st.spinner("Searching similar locations..."):
                try:
                    from clustering_discovery import find_similar_sites
                    similar_results = find_similar_sites(chosen_tile, top_k=6)
                    if similar_results:
                        st.success(f"{len(similar_results)} similar locations discovered.")
                        cols = st.columns(len(similar_results))
                        from semantic_search import tile_to_rgb
                        for idx, res in enumerate(similar_results):
                            with cols[idx]:
                                st.markdown(f"**Match #{res['rank']}**")
                                st.caption(f"Similarity: **{res['similarity_score']:.4f}**")
                                st.caption(f"Date: `{res['date']}`")
                                t_path = os.path.join(TILES_DIR, res["tile_file"])
                                if os.path.exists(t_path):
                                    try:
                                        img = tile_to_rgb(t_path)
                                        st.image(img, use_container_width=True, caption=res["tile_file"])
                                    except Exception:
                                        st.code(res["tile_file"])
                    else:
                        st.info("No matching locations found above similarity threshold.")
                except Exception as e:
                    st.error(f"Error: {e}")

    with subtab2:
        st.subheader("8 Unsupervised Terrain Archetypes (KMeans k=8 Clustering)")
        if os.path.exists(CLUSTERS_FILE):
            with open(CLUSTERS_FILE, "r", encoding="utf-8") as f:
                c_data = json.load(f)

            summary_list = []
            for c_id, info in c_data.items():
                summary_list.append({
                    "Cluster ID": c_id,
                    "Total Tiles": info["total_tiles"],
                    "Example Tile": info["exemplar_tile"]
                })
            st.dataframe(pd.DataFrame(summary_list), use_container_width=True)
        else:
            st.info("Cluster index not yet generated. Initialize below:")
            if st.button("Initialize Clustering Pipeline"):
                os.system("python CODE/clustering_discovery.py --build_clusters")
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: ANALYST DECISIONS
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("📑 Analyst Decisions")
    st.caption("Comprehensive audit trail and forensic verification log of analyst decisions")

    decisions = load_decisions()
    if decisions:
        df_dec = pd.DataFrame(decisions)
        st.subheader(f"Total {len(df_dec)} decisions recorded")
        st.dataframe(df_dec, use_container_width=True)

        csv_data = df_dec.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Decisions (CSV)",
            data=csv_data,
            file_name="analyst_decisions.csv",
            mime="text/csv"
        )
    else:
        st.info("No analyst decisions recorded yet. Review pending candidates in the Change Review Queue.")

    st.divider()
    st.subheader("📊 System Info")
    eval_metrics = {
        "Location": "Ranchi Urban & Peri-Urban, Jharkhand, India",
        "Bounding Box": "85.15°E–85.45°E, 23.20°N–23.45°N (~868 km²)",
        "Time Period": "2020-01-01 to 2024-12-31 (6 time points)",
        "Satellite": "Sentinel-2 MSI Level-2A (10m resolution, 4 bands)",
        "Total Tiles": 180,
        "Embedding Dimensions": 512,
        "Search Index": "FAISS IndexFlatIP (Cosine Similarity)",
        "Change Detection": "Multi-Temporal Vector Distance + Spectral Morphology",
        "False Alarm Filter": "Seasonal Mismatch Penalty + Quality Masking"
    }
    st.json(eval_metrics)
