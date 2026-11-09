import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ==============================================================================
# CARDIOSCAN — REAL 3D HUMAN HEART TELEMETRY
# Real Anatomical 3D Model from 'human-heart/source/Human heart.glb'
# Pure White (#FFFFFF) | Deep Charcoal (#111112) | Distinct Muted Class Palettes
# Zero Neon | Zero Emojis | Full GUI Class Tinting
# ==============================================================================
st.set_page_config(
    page_title="CardioScan",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Streamlit Clean Frame
st.markdown("""
<style>
    #MainMenu, header, footer {visibility: hidden; height: 0px !important;}
    .block-container {
        padding-top: 0.4rem !important;
        padding-bottom: 0.6rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
        max-width: 1400px !important;
    }
    .stApp {
        background-color: #FFFFFF !important;
        color: #111112 !important;
    }
</style>
""", unsafe_allow_html=True)

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

curated_file = APP_DIR / "curated_beats.json"
weights_file = APP_DIR / "model_weights.json"
heart_mesh_file = APP_DIR / "human_heart_mesh.json"

if not curated_file.exists() or not weights_file.exists() or not heart_mesh_file.exists():
    st.error("Missing asset files in app directory.")
    st.stop()

with open(curated_file, "r") as f:
    curated_beats = json.load(f)

with open(weights_file, "r") as f:
    model_weights = json.load(f)

with open(heart_mesh_file, "r") as f:
    heart_mesh_data = json.load(f)

curated_json_str = json.dumps(curated_beats)
weights_json_str = json.dumps(model_weights)
heart_mesh_json_str = json.dumps(heart_mesh_data)

# Main Application HTML/CSS/JS (Real 3D Human Heart from GLB, 100% Offline Canvas Engine)
app_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CardioScan</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    :root {{
        --bg-white: #FFFFFF;
        --bg-subtle: #F8F9FA;
        --bg-surface: #FAFAFC;
        --text-primary: #111112;
        --text-secondary: #52525B;
        --text-muted: #71717A;
        --border: #E4E4E7;
        --border-hover: #111112;
        
        /* Dynamic Theme Active State (Defaults to Class 0 Blue) */
        --theme-color: #2563EB;
        --theme-rgb: 37, 99, 235;
        --theme-bg-soft: rgba(37, 99, 235, 0.06);
        --theme-border-soft: rgba(37, 99, 235, 0.25);
        
        --shadow: 0 1px 3px rgba(0, 0, 0, 0.04), 0 14px 30px rgba(0, 0, 0, 0.03);
    }}

    * {{
        box-sizing: border-box;
        margin: 0;
        padding: 0;
        user-select: none;
    }}

    body {{
        background: var(--bg-white);
        color: var(--text-primary);
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        padding: 6px 10px;
        -webkit-font-smoothing: antialiased;
        transition: background 0.4s ease;
    }}

    .app-shell {{
        max-width: 1360px;
        margin: 0 auto;
        display: flex;
        flex-direction: column;
        gap: 14px;
    }}

    /* Minimal Top Bar (No Audio / Mute Button) */
    .top-bar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 18px;
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 12px;
    }}

    .brand-group {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 800;
        font-size: 1.15rem;
        letter-spacing: -0.02em;
    }}

    .brand-icon {{
        width: 18px;
        height: 18px;
        fill: none;
        stroke: var(--theme-color);
        stroke-width: 2.2;
        stroke-linecap: round;
        stroke-linejoin: round;
        transition: stroke 0.4s ease;
    }}

    .team-capsule {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        color: var(--text-secondary);
        background: var(--bg-subtle);
        border: 1px solid var(--border);
        padding: 6px 14px;
        border-radius: 9999px;
    }}
    .team-capsule strong {{
        color: var(--text-primary);
    }}

    /* =========================================================================
       HERO BANNER (3D OBJECT STAGE)
       ========================================================================= */
    .hero-stage {{
        display: grid;
        grid-template-columns: 1.2fr 0.8fr;
        gap: 20px;
        background: linear-gradient(180deg, #FFFFFF 0%, var(--bg-surface) 100%);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 24px 30px;
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
        transition: border-color 0.4s ease, box-shadow 0.4s ease;
    }}

    /* Dynamic Ambient Tint inside the Hero Stage */
    .hero-ambient-tint {{
        position: absolute;
        top: 0;
        right: 0;
        width: 50%;
        height: 100%;
        background: radial-gradient(circle at 70% 50%, var(--theme-bg-soft) 0%, transparent 70%);
        pointer-events: none;
        transition: background 0.6s ease;
    }}

    .hero-content {{
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 12px;
        z-index: 2;
    }}

    .hero-headline {{
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.15;
        letter-spacing: -0.02em;
    }}
    .hero-headline span {{
        color: var(--text-muted);
        font-weight: 400;
    }}

    /* 3 Mode Selection Tabs */
    .mode-tab-bar {{
        display: flex;
        gap: 6px;
        background: var(--bg-subtle);
        border: 1px solid var(--border);
        padding: 4px;
        border-radius: 10px;
        width: fit-content;
    }}

    .mode-tab {{
        background: transparent;
        border: none;
        padding: 6px 14px;
        border-radius: 7px;
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--text-secondary);
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 6px;
        transition: all 0.15s;
    }}

    .mode-tab.active {{
        background: #FFFFFF;
        color: var(--text-primary);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }}

    /* Mode Panels */
    .mode-panel {{
        display: none;
        flex-direction: column;
        gap: 10px;
        margin-top: 4px;
    }}
    .mode-panel.visible {{
        display: flex;
    }}

    .action-row {{
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }}

    .btn-action {{
        background: var(--text-primary);
        color: #FFFFFF;
        border: 1px solid var(--text-primary);
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        transition: all 0.15s;
    }}
    .btn-action:hover {{
        background: #27272A;
    }}

    .btn-outline {{
        background: #FFFFFF;
        color: var(--text-primary);
        border: 1px solid var(--border);
        padding: 7px 13px;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 600;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }}
    .btn-outline:hover {{
        border-color: var(--text-primary);
    }}

    .btn-outline.active {{
        background: var(--theme-color) !important;
        color: #FFFFFF !important;
        border-color: var(--theme-color) !important;
        box-shadow: 0 2px 8px var(--theme-bg-soft);
    }}

    .custom-input-box {{
        width: 100%;
        height: 60px;
        padding: 8px 10px;
        border: 1px solid var(--border);
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: var(--text-primary);
        background: #FFFFFF;
        resize: none;
        outline: none;
    }}
    .custom-input-box:focus {{
        border-color: var(--theme-color);
    }}

    .file-drop-area {{
        border: 1.5px dashed var(--border);
        border-radius: 10px;
        padding: 12px 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #FFFFFF;
        cursor: pointer;
    }}
    .file-drop-area:hover {{
        border-color: var(--theme-color);
    }}

    /* 3D Heart Stage */
    .hero-3d-stage {{
        position: relative;
        height: 250px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: grab;
        z-index: 2;
    }}
    .hero-3d-stage:active {{
        cursor: grabbing;
    }}

    canvas#canvas3D {{
        width: 100%;
        height: 100%;
        display: block;
    }}

    /* Dominant Ground Contact Shadow */
    .stage-shadow-disc {{
        position: absolute;
        bottom: 6px;
        width: 190px;
        height: 26px;
        background: radial-gradient(ellipse at 50% 50%, rgba(17, 17, 18, 0.32) 0%, rgba(17, 17, 18, 0.14) 42%, transparent 75%);
        border-radius: 50%;
        pointer-events: none;
        transition: transform 0.25s ease, opacity 0.25s ease;
    }}

    /* =========================================================================
       RESULTS & TELEMETRY SECTION
       ========================================================================= */
    .cockpit-row {{
        display: grid;
        grid-template-columns: 1.35fr 1fr;
        gap: 16px;
    }}

    .telemetry-card {{
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px 22px;
        box-shadow: var(--shadow);
        display: flex;
        flex-direction: column;
        gap: 12px;
        position: relative;
        transition: border-color 0.4s ease;
    }}

    .card-top {{
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}

    .card-lbl {{
        font-size: 0.86rem;
        font-weight: 700;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 6px;
    }}

    .scope-frame {{
        width: 100%;
        height: 195px;
        background: #FAFAFC;
        border: 1px solid var(--border);
        border-radius: 10px;
        position: relative;
        overflow: hidden;
    }}

    canvas#scopeCanvas {{
        width: 100%;
        height: 100%;
        display: block;
    }}

    /* Calculating Holographic Pulse Overlay */
    .calc-scan-overlay {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(2px);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 8px;
        z-index: 10;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.2s ease;
    }}
    .calc-scan-overlay.active {{
        opacity: 1;
        pointer-events: all;
    }}

    .scan-beam {{
        width: 38px;
        height: 38px;
        border: 3px solid #E4E4E7;
        border-top-color: var(--theme-color);
        border-radius: 50%;
        animation: spin 0.6s linear infinite;
    }}

    @keyframes spin {{
        to {{ transform: rotate(360deg); }}
    }}

    /* Decision Metric Capsule */
    .decision-capsule {{
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 12px 16px;
        background: var(--theme-bg-soft);
        border: 1px solid var(--theme-border-soft);
        border-radius: 12px;
        transition: background 0.4s ease, border-color 0.4s ease;
    }}

    .gauge-box {{
        position: relative;
        width: 72px;
        height: 72px;
        flex-shrink: 0;
    }}

    .gauge-svg {{
        width: 100%;
        height: 100%;
        transform: rotate(-90deg);
    }}

    .gauge-circle-bg {{
        fill: none;
        stroke: #E4E4E7;
        stroke-width: 7;
    }}

    .gauge-circle-fg {{
        fill: none;
        stroke: var(--theme-color);
        stroke-width: 7;
        stroke-linecap: round;
        stroke-dasharray: 188;
        stroke-dashoffset: 10;
        transition: stroke-dashoffset 0.4s ease, stroke 0.4s ease;
    }}

    .gauge-mid-text {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        font-weight: 800;
        color: var(--text-primary);
    }}

    .decision-meta {{
        display: flex;
        flex-direction: column;
        gap: 3px;
    }}

    .class-badge {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 5px;
        width: fit-content;
        letter-spacing: 0.03em;
        background: #FFFFFF;
        color: var(--theme-color);
        border: 1px solid var(--theme-border-soft);
        transition: color 0.4s ease, border-color 0.4s ease;
    }}

    .class-name {{
        font-size: 1.05rem;
        font-weight: 800;
        color: var(--text-primary);
    }}

    /* Minimal Probability Bars */
    .prob-list {{
        display: flex;
        flex-direction: column;
        gap: 7px;
    }}

    .prob-item {{
        display: flex;
        flex-direction: column;
        gap: 3px;
    }}

    .prob-header {{
        display: flex;
        justify-content: space-between;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
    }}

    .prob-track {{
        width: 100%;
        height: 5px;
        background: #F1F5F9;
        border-radius: 9999px;
        overflow: hidden;
    }}

    .prob-fill {{
        height: 100%;
        background: #CBD5E1;
        border-radius: 9999px;
        transition: width 0.35s ease, background 0.3s ease;
    }}
    .prob-fill.lead {{
        background: var(--theme-color) !important;
    }}

    /* Inline SVG Icon Helper */
    .icon-svg {{
        width: 14px;
        height: 14px;
        stroke: currentColor;
        stroke-width: 2;
        fill: none;
        stroke-linecap: round;
        stroke-linejoin: round;
        display: inline-block;
        vertical-align: middle;
    }}
</style>
</head>
<body>

<div class="app-shell">

    <!-- Top Navigation Bar -->
    <nav class="top-bar">
        <div class="brand-group">
            <svg class="brand-icon" viewBox="0 0 24 24">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
            </svg>
            <span>CardioScan</span>
        </div>

        <div class="team-capsule">
            <strong>Rizwana Mubarak (23F-6055)</strong> &bull; <strong>Abdul Rehman (23F-6075)</strong>
        </div>
    </nav>

    <!-- 3D Hero Banner with Real Human Heart Model -->
    <section class="hero-stage">
        <div class="hero-ambient-tint" id="ambientTint"></div>

        <div class="hero-content">
            <h1 class="hero-headline">
                Cardiac Arrhythmia <span>Classifier</span>
            </h1>

            <!-- 3 Mode Selection Tabs -->
            <div class="mode-tab-bar">
                <button class="mode-tab active" id="tab_rnd" onclick="switchMode('rnd')">
                    <svg class="icon-svg" viewBox="0 0 24 24"><polyline points="16 3 21 3 21 8"></polyline><line x1="4" y1="20" x2="21" y2="3"></line><polyline points="21 16 21 21 16 21"></polyline><line x1="15" y1="15" x2="21" y2="21"></line><line x1="4" y1="4" x2="9" y2="9"></line></svg>
                    Random Generator
                </button>
                <button class="mode-tab" id="tab_manual" onclick="switchMode('manual')">
                    <svg class="icon-svg" viewBox="0 0 24 24"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>
                    Manual Input
                </button>
                <button class="mode-tab" id="tab_csv" onclick="switchMode('csv')">
                    <svg class="icon-svg" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                    CSV File
                </button>
            </div>

            <!-- Option 1: Random Generator Panel -->
            <div class="mode-panel visible" id="panel_rnd">
                <div class="action-row">
                    <button class="btn-action" onclick="generateRandomBeat()">
                        <svg class="icon-svg" viewBox="0 0 24 24"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"></path></svg>
                        Generate Random Beat
                    </button>
                    <button class="btn-outline active" id="btnP_0" onclick="selectPresetClass(0)">Normal [N]</button>
                    <button class="btn-outline" id="btnP_1" onclick="selectPresetClass(1)">Supravent. [S]</button>
                    <button class="btn-outline" id="btnP_2" onclick="selectPresetClass(2)">Ventricular [V]</button>
                    <button class="btn-outline" id="btnP_3" onclick="selectPresetClass(3)">Fusion [F]</button>
                    <button class="btn-outline" id="btnP_4" onclick="selectPresetClass(4)">Paced [Q]</button>
                </div>
            </div>

            <!-- Option 2: Manual Input Panel -->
            <div class="mode-panel" id="panel_manual">
                <textarea class="custom-input-box" id="manualInputText" placeholder="Enter comma-separated 187 float values (0.0 to 1.0)..."></textarea>
                <div class="action-row">
                    <button class="btn-action" onclick="predictManualInput()">
                        <svg class="icon-svg" viewBox="0 0 24 24"><polyline points="9 11 12 14 22 4"></polyline><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"></path></svg>
                        Compute Prediction
                    </button>
                    <button class="btn-outline" onclick="loadTemplateToManual()">Load Template</button>
                    <button class="btn-outline" onclick="clearManualInput()">Clear</button>
                </div>
            </div>

            <!-- Option 3: CSV File Panel -->
            <div class="mode-panel" id="panel_csv">
                <div class="file-drop-area" onclick="document.getElementById('csvFileInput').click()">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <svg class="icon-svg" style="width: 20px; height: 20px;" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
                        <div>
                            <div style="font-size: 0.82rem; font-weight: 700;" id="csvFileName">Click to select CSV File</div>
                            <div style="font-size: 0.72rem; color: var(--text-muted);" id="csvStatus">Supports MIT-BIH row formats (187 or 188 columns)</div>
                        </div>
                    </div>
                    <button class="btn-outline" style="pointer-events: none;">Select File</button>
                    <input type="file" id="csvFileInput" accept=".csv" style="display: none;" onchange="handleCsvFile(this)">
                </div>
                <div class="action-row" id="csvNavRow" style="display: none;">
                    <span style="font-size: 0.76rem; font-family: 'JetBrains Mono'; color: var(--text-muted);" id="csvRowCount">Row 1 of 0</span>
                    <button class="btn-outline" onclick="stepCsvRow(-1)">Previous Row</button>
                    <button class="btn-outline" onclick="stepCsvRow(1)">Next Row</button>
                    <button class="btn-action" onclick="predictCurrentCsvRow()">Analyze Row</button>
                </div>
            </div>

        </div>

        <!-- Real 3D Human Heart Stage -->
        <div class="hero-3d-stage" id="stage3D">
            <canvas id="canvas3D"></canvas>
            <div class="stage-shadow-disc" id="shadowDisc"></div>
        </div>

    </section>

    <!-- Results Section -->
    <div class="cockpit-row">

        <!-- Oscilloscope Lead -->
        <div class="telemetry-card">
            <div class="calc-scan-overlay" id="calcOverlay">
                <div class="scan-beam"></div>
                <span style="font-family: 'JetBrains Mono'; font-size: 0.76rem; font-weight: 600;">Evaluating Scratch Tensor Network...</span>
            </div>

            <div class="card-top">
                <div class="card-lbl">
                    <svg class="icon-svg" viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
                    <span>Vector Oscilloscope</span>
                </div>
                <div style="font-family: 'JetBrains Mono'; font-size: 0.74rem; color: var(--text-muted);">
                    R-PEAK: <strong id="peakVal" style="color: var(--text-primary);">0.982 V</strong>
                </div>
            </div>

            <div class="scope-frame">
                <canvas id="scopeCanvas"></canvas>
            </div>
        </div>

        <!-- Decision & Spectrum -->
        <div class="telemetry-card">
            <div class="card-top">
                <div class="card-lbl">
                    <svg class="icon-svg" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 14 14"></polyline></svg>
                    <span>Classification Result</span>
                </div>
                <div style="font-family: 'JetBrains Mono'; font-size: 0.72rem; color: var(--text-muted);">&lt; 0.1ms Forward Pass</div>
            </div>

            <!-- Decision Capsule -->
            <div class="decision-capsule">
                <div class="gauge-box">
                    <svg class="gauge-svg" viewBox="0 0 70 70">
                        <circle class="gauge-circle-bg" cx="35" cy="35" r="30"></circle>
                        <circle class="gauge-circle-fg" id="gaugeCircle" cx="35" cy="35" r="30"></circle>
                    </svg>
                    <div class="gauge-mid-text" id="gaugeNum">99.8%</div>
                </div>

                <div class="decision-meta">
                    <div class="class-badge normal" id="badgePill">CLASS 0 &bull; NORMAL</div>
                    <div class="class-name" id="decisionName">Normal Sinus Rhythm</div>
                </div>
            </div>

            <!-- Probability Bars -->
            <div class="prob-list">
                <div class="prob-item">
                    <div class="prob-header">
                        <span>Normal (N)</span>
                        <strong id="pVal_0">99.8%</strong>
                    </div>
                    <div class="prob-track"><div class="prob-fill lead" id="pFill_0" style="width: 99.8%;"></div></div>
                </div>
                <div class="prob-item">
                    <div class="prob-header">
                        <span>Supraventricular (S)</span>
                        <strong id="pVal_1">0.1%</strong>
                    </div>
                    <div class="prob-track"><div class="prob-fill" id="pFill_1" style="width: 1%;"></div></div>
                </div>
                <div class="prob-item">
                    <div class="prob-header">
                        <span>Ventricular (V)</span>
                        <strong id="pVal_2">0.1%</strong>
                    </div>
                    <div class="prob-track"><div class="prob-fill" id="pFill_2" style="width: 1%;"></div></div>
                </div>
                <div class="prob-item">
                    <div class="prob-header">
                        <span>Fusion (F)</span>
                        <strong id="pVal_3">0.0%</strong>
                    </div>
                    <div class="prob-track"><div class="prob-fill" id="pFill_3" style="width: 1%;"></div></div>
                </div>
                <div class="prob-item">
                    <div class="prob-header">
                        <span>Paced (Q)</span>
                        <strong id="pVal_4">0.0%</strong>
                    </div>
                    <div class="prob-track"><div class="prob-fill" id="pFill_4" style="width: 1%;"></div></div>
                </div>
            </div>

        </div>

    </div>

</div>

<script>
    const CURATED = {curated_json_str};
    const MODEL = {weights_json_str};
    const HEART_MESH = {heart_mesh_json_str};

    // 5 Distinct, Soft, Natural Class Colors (Zero Neon)
    const CLASS_PALETTES = {{
        0: {{ name: "Normal Sinus Rhythm", code: "CLASS 0 • NORMAL", color: "#2563EB", rgb: "37, 99, 235" }},
        1: {{ name: "Supraventricular Ectopic", code: "CLASS 1 • SUPRAVENT.", color: "#D97706", rgb: "217, 119, 6" }},
        2: {{ name: "Ventricular Ectopic (PVC)", code: "CLASS 2 • VENTRICULAR", color: "#DC2626", rgb: "220, 38, 38" }},
        3: {{ name: "Fusion Beat", code: "CLASS 3 • FUSION", color: "#7C3AED", rgb: "124, 58, 237" }},
        4: {{ name: "Paced / Unknown Beat", code: "CLASS 4 • PACED", color: "#059669", rgb: "5, 150, 105" }}
    }};

    let currentSignal = new Float32Array(CURATED[0][0].signal);
    let activeClass = 0;
    let csvData = [];
    let currentCsvRow = 0;

    // 3D Heart Color Transition: 0 = Pure Technical Sketch, 1 = Rich Colorized
    let heartColorIntensity = 0.0;
    let activeRgb = [37, 99, 235];
    let targetRgb = [37, 99, 235];

    // Decode Base64 Binary Geometry from Human Heart GLB
    function decodeBase64Float32(b64) {{
        const bin = atob(b64);
        const len = bin.length;
        const bytes = new Uint8Array(len);
        for(let i=0; i<len; i++) bytes[i] = bin.charCodeAt(i);
        return new Float32Array(bytes.buffer);
    }}

    function decodeBase64Uint16(b64) {{
        const bin = atob(b64);
        const len = bin.length;
        const bytes = new Uint8Array(len);
        for(let i=0; i<len; i++) bytes[i] = bin.charCodeAt(i);
        return new Uint16Array(bytes.buffer);
    }}

    const rawHeartPositions = decodeBase64Float32(HEART_MESH.positions);
    const rawHeartIndices = decodeBase64Uint16(HEART_MESH.indices);
    const rawFeatureEdges = decodeBase64Uint16(HEART_MESH.feature_edges);
    const numHeartVerts = HEART_MESH.num_vertices;
    const numHeartTris = HEART_MESH.num_triangles;
    const numFeatureEdges = HEART_MESH.num_feature_edges;

    // Propagate Detected Class Color Across Entire GUI
    function applyClassThemeToGUI(classId) {{
        activeClass = classId;
        const pal = CLASS_PALETTES[classId];
        targetRgb = pal.rgb.split(",").map(v => parseInt(v.trim()));

        document.documentElement.style.setProperty("--theme-color", pal.color);
        document.documentElement.style.setProperty("--theme-rgb", pal.rgb);
        document.documentElement.style.setProperty("--theme-bg-soft", `rgba(${{pal.rgb}}, 0.08)`);
        document.documentElement.style.setProperty("--theme-border-soft", `rgba(${{pal.rgb}}, 0.28)`);

        for(let i=0; i<5; i++) {{
            const btn = document.getElementById(`btnP_${{i}}`);
            if (btn) {{
                if (i === classId) btn.classList.add("active");
                else btn.classList.remove("active");
            }}
        }}

        // Smoothly bloom into predicted class's color
        heartColorIntensity = 1.0;
    }}

    function switchMode(mode) {{
        document.getElementById("tab_rnd").className = mode === "rnd" ? "mode-tab active" : "mode-tab";
        document.getElementById("tab_manual").className = mode === "manual" ? "mode-tab active" : "mode-tab";
        document.getElementById("tab_csv").className = mode === "csv" ? "mode-tab active" : "mode-tab";

        document.getElementById("panel_rnd").className = mode === "rnd" ? "mode-panel visible" : "mode-panel";
        document.getElementById("panel_manual").className = mode === "manual" ? "mode-panel visible" : "mode-panel";
        document.getElementById("panel_csv").className = mode === "csv" ? "mode-panel visible" : "mode-panel";
    }}

    // Scratch Forward Pass (Zero external libraries)
    function predictScratch(x) {{
        const W1 = MODEL.W1, b1 = MODEL.b1;
        const W2 = MODEL.W2, b2 = MODEL.b2;
        const W3 = MODEL.W3, b3 = MODEL.b3;

        const z1 = new Float32Array(128);
        for(let i=0; i<128; i++) {{
            let s = b1[i];
            const r = W1[i];
            for(let j=0; j<187; j++) s += r[j] * x[j];
            z1[i] = s > 0 ? s : 0;
        }}

        const z2 = new Float32Array(64);
        for(let i=0; i<64; i++) {{
            let s = b2[i];
            const r = W2[i];
            for(let j=0; j<128; j++) s += r[j] * z1[j];
            z2[i] = s > 0 ? s : 0;
        }}

        const z3 = new Float32Array(5);
        let maxZ = -Infinity;
        for(let i=0; i<5; i++) {{
            let s = b3[i];
            const r = W3[i];
            for(let j=0; j<64; j++) s += r[j] * z2[j];
            z3[i] = s;
            if(s > maxZ) maxZ = s;
        }}

        const probs = new Float32Array(5);
        let sumExp = 0;
        for(let i=0; i<5; i++) {{
            probs[i] = Math.exp(z3[i] - maxZ);
            sumExp += probs[i];
        }}
        for(let i=0; i<5; i++) probs[i] /= sumExp;
        return probs;
    }}

    // Animated Prediction Execution
    function executePredictionWithAnimation(signalArray) {{
        const overlay = document.getElementById("calcOverlay");
        overlay.classList.add("active");

        setTimeout(() => {{
            overlay.classList.remove("active");
            currentSignal = new Float32Array(signalArray);

            const probs = predictScratch(currentSignal);
            let pred = 0, bestP = 0;
            for(let i=0; i<5; i++) {{
                if(probs[i] > bestP) {{
                    bestP = probs[i];
                    pred = i;
                }}
            }}

            const pal = CLASS_PALETTES[pred];
            const confPct = (bestP * 100).toFixed(1);

            applyClassThemeToGUI(pred);

            // Arc Gauge
            const circle = document.getElementById("gaugeCircle");
            const circumference = 188;
            const offset = circumference - (bestP * circumference);
            circle.style.strokeDashoffset = offset;
            document.getElementById("gaugeNum").textContent = `${{confPct}}%`;

            // Badge & Title
            const badge = document.getElementById("badgePill");
            badge.textContent = pal.code;
            document.getElementById("decisionName").textContent = pal.name;

            // Probability Bars
            for(let k=0; k<5; k++) {{
                const p = (probs[k] * 100).toFixed(1);
                document.getElementById(`pVal_${{k}}`).textContent = `${{p}}%`;
                const fill = document.getElementById(`pFill_${{k}}`);
                fill.style.width = `${{Math.max(p, 1)}}%`;

                if (k === pred) {{
                    fill.className = "prob-fill lead";
                    document.getElementById(`pVal_${{k}}`).style.color = pal.color;
                    document.getElementById(`pVal_${{k}}`).style.fontWeight = "700";
                }} else {{
                    fill.className = "prob-fill";
                    document.getElementById(`pVal_${{k}}`).style.color = "#71717A";
                    document.getElementById(`pVal_${{k}}`).style.fontWeight = "400";
                }}
            }}

            triggerContractionPulse();
            renderOscilloscope();
        }}, 360);
    }}

    // Option 1: Random Generation
    function generateRandomBeat() {{
        const rndClass = Math.floor(Math.random() * 5);
        selectPresetClass(rndClass);
    }}

    function selectPresetClass(c) {{
        const pool = CURATED[c];
        const pick = pool[Math.floor(Math.random() * pool.length)].signal;

        const perturbed = new Float32Array(187);
        for(let i=0; i<187; i++) {{
            perturbed[i] = Math.max(0, Math.min(1.2, pick[i] + (Math.random() - 0.5) * 0.025));
        }}

        executePredictionWithAnimation(perturbed);
    }}

    // Option 2: Manual Input
    function predictManualInput() {{
        const raw = document.getElementById("manualInputText").value;
        const parts = raw.split(/[,\\s\\t\\n]+/).filter(v => v.trim().length > 0);
        if (parts.length === 0) {{
            alert("Please enter comma-separated ECG values.");
            return;
        }}

        const vec = new Float32Array(187);
        for(let i=0; i<187; i++) {{
            if (i < parts.length) {{
                const val = parseFloat(parts[i]);
                vec[i] = isNaN(val) ? 0 : Math.max(0, Math.min(1.5, val));
            }} else {{
                vec[i] = 0;
            }}
        }}

        executePredictionWithAnimation(vec);
    }}

    function loadTemplateToManual() {{
        const sample = CURATED[activeClass][0].signal;
        document.getElementById("manualInputText").value = sample.map(v => v.toFixed(4)).join(", ");
    }}

    function clearManualInput() {{
        document.getElementById("manualInputText").value = "";
    }}

    // Option 3: CSV File Handling
    function handleCsvFile(input) {{
        const file = input.files[0];
        if (!file) return;

        document.getElementById("csvFileName").textContent = file.name;
        document.getElementById("csvStatus").textContent = "Parsing CSV records...";

        const reader = new FileReader();
        reader.onload = function(e) {{
            const lines = e.target.result.split(/\\r?\\n/).filter(l => l.trim().length > 0);
            csvData = [];
            for(let line of lines) {{
                const nums = line.split(",").map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
                if (nums.length >= 187) {{
                    csvData.push(nums.slice(0, 187));
                }}
            }}

            if (csvData.length > 0) {{
                currentCsvRow = 0;
                document.getElementById("csvNavRow").style.display = "flex";
                updateCsvStatus();
                predictCurrentCsvRow();
            }} else {{
                document.getElementById("csvStatus").textContent = "No valid 187-column rows found.";
            }}
        }};
        reader.readAsText(file);
    }}

    function updateCsvStatus() {{
        document.getElementById("csvRowCount").textContent = `Row ${{currentCsvRow + 1}} of ${{csvData.length}}`;
    }}

    function stepCsvRow(dir) {{
        if (csvData.length === 0) return;
        currentCsvRow = Math.max(0, Math.min(csvData.length - 1, currentCsvRow + dir));
        updateCsvStatus();
        predictCurrentCsvRow();
    }}

    function predictCurrentCsvRow() {{
        if (csvData.length === 0) return;
        executePredictionWithAnimation(csvData[currentCsvRow]);
    }}

    // =========================================================================
    // 3D REAL HUMAN HEART ENGINE (From Human heart.glb)
    // Zero External Libs • Pure Canvas 3D Perspective Projection & Shading
    // =========================================================================
    const canvas3D = document.getElementById("canvas3D");
    const ctx3D = canvas3D.getContext("2d");

    let rotX = 0.15, rotY = 0.40;
    let targetRotX = 0.15, targetRotY = 0.40;
    let isDragging = false;
    let lastMouseX = 0, lastMouseY = 0;
    let pulseScale = 1.0;

    function triggerContractionPulse() {{
        pulseScale = 1.18;
    }}

    const stageWrap = document.getElementById("stage3D");
    stageWrap.addEventListener("mousedown", (e) => {{
        isDragging = true;
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
    }});
    window.addEventListener("mouseup", () => {{ isDragging = false; }});
    window.addEventListener("mousemove", (e) => {{
        if (isDragging) {{
            const dx = e.clientX - lastMouseX, dy = e.clientY - lastMouseY;
            targetRotY += dx * 0.012;
            targetRotX += dy * 0.012;
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
        }}
    }});

    // Projected buffer
    const projX = new Float32Array(numHeartVerts);
    const projY = new Float32Array(numHeartVerts);
    const projZ = new Float32Array(numHeartVerts);

    // Pre-allocated face sort array
    const sortedFaces = new Array(numHeartTris);
    for(let f=0; f<numHeartTris; f++) {{
        sortedFaces[f] = {{ i1: 0, i2: 0, i3: 0, avgZ: 0, crossZ: 0 }};
    }}

    function render3D() {{
        const rect = canvas3D.getBoundingClientRect();
        const w = rect.width, h = rect.height;

        if (canvas3D.width !== w * window.devicePixelRatio) {{
            canvas3D.width = w * window.devicePixelRatio;
            canvas3D.height = h * window.devicePixelRatio;
            ctx3D.scale(window.devicePixelRatio, window.devicePixelRatio);
        }}

        ctx3D.clearRect(0, 0, w, h);

        rotX += (targetRotX - rotX) * 0.08;
        rotY += (targetRotY - rotY) * 0.08;
        pulseScale += (1.0 - pulseScale) * 0.12;

        activeRgb[0] += (targetRgb[0] - activeRgb[0]) * 0.10;
        activeRgb[1] += (targetRgb[1] - activeRgb[1]) * 0.10;
        activeRgb[2] += (targetRgb[2] - activeRgb[2]) * 0.10;

        // 10% Zoom out from 305 -> 275
        const fov = 275, dist = 2.4;
        const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
        const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
        const cx = w / 2, cy = h / 2 - 8;

        // Project 100% full original vertices from the GLB geometry
        for (let i = 0; i < numHeartVerts; i++) {{
            const base = i * 3;
            let vx = rawHeartPositions[base] * pulseScale;
            let vy = -rawHeartPositions[base + 1] * pulseScale;
            let vz = rawHeartPositions[base + 2] * pulseScale;

            // Y Rotation
            const x1 = vx * cosY - vz * sinY;
            const z1 = vx * sinY + vz * cosY;

            // X Rotation
            const y2 = vy * cosX - z1 * sinX;
            const z2 = vy * sinX + z1 * cosX;

            const scale = fov / (z2 + dist);
            projX[i] = cx + x1 * scale;
            projY[i] = cy + y2 * scale;
            projZ[i] = z2;
        }}

        // Backface culling & depth accumulation
        let visibleCount = 0;
        for (let f = 0; f < numHeartTris; f++) {{
            const base = f * 3;
            const i1 = rawHeartIndices[base];
            const i2 = rawHeartIndices[base + 1];
            const i3 = rawHeartIndices[base + 2];

            const ax = projX[i2] - projX[i1], ay = projY[i2] - projY[i1];
            const bx = projX[i3] - projX[i1], by = projY[i3] - projY[i1];
            const crossZ = ax * by - ay * bx;

            if (crossZ > 0) {{
                const faceObj = sortedFaces[visibleCount];
                faceObj.i1 = i1;
                faceObj.i2 = i2;
                faceObj.i3 = i3;
                faceObj.avgZ = (projZ[i1] + projZ[i2] + projZ[i3]) / 3;
                faceObj.crossZ = crossZ;
                visibleCount++;
            }}
        }}

        // Painter's sorting (back-to-front)
        const activeSublist = sortedFaces.slice(0, visibleCount);
        activeSublist.sort((a, b) => b.avgZ - a.avgZ);

        // Render 100% Exact Human Heart Shaded Surfaces
        for (let k = 0; k < visibleCount; k++) {{
            const face = activeSublist[k];
            const i1 = face.i1, i2 = face.i2, i3 = face.i3;

            // Soft directional lighting from top-left
            const light = Math.min(1.0, Math.max(0.18, (face.crossZ / 26.0)));

            // Pure architectural medical tone
            const sketchGray = Math.floor(250 - light * 34);

            // Colored Mode using detected class palette
            const colR = Math.floor(activeRgb[0] * (0.40 + light * 0.60));
            const colG = Math.floor(activeRgb[1] * (0.40 + light * 0.60));
            const colB = Math.floor(activeRgb[2] * (0.40 + light * 0.60));

            const rFinal = Math.floor(sketchGray * (1.0 - heartColorIntensity) + colR * heartColorIntensity);
            const gFinal = Math.floor(sketchGray * (1.0 - heartColorIntensity) + colG * heartColorIntensity);
            const bFinal = Math.floor(sketchGray * (1.0 - heartColorIntensity) + colB * heartColorIntensity);

            ctx3D.beginPath();
            ctx3D.moveTo(projX[i1], projY[i1]);
            ctx3D.lineTo(projX[i2], projY[i2]);
            ctx3D.lineTo(projX[i3], projY[i3]);
            ctx3D.closePath();

            ctx3D.fillStyle = `rgb(${{rFinal}}, ${{gFinal}}, ${{bFinal}})`;
            ctx3D.fill();
        }}

        // Draw Anatomical Contour & Feature Crease Lines (Aorta, sulcus, coronary grooves)
        if (numFeatureEdges > 0) {{
            ctx3D.beginPath();
            const edgeAlpha = heartColorIntensity < 0.6 ? 0.32 : 0.18;
            ctx3D.strokeStyle = heartColorIntensity < 0.6 ? `rgba(17, 17, 18, ${{edgeAlpha}})` : `rgba(255, 255, 255, ${{edgeAlpha}})`;
            ctx3D.lineWidth = 0.8;

            for (let e = 0; e < numFeatureEdges; e++) {{
                const base = e * 2;
                const v1 = rawFeatureEdges[base];
                const v2 = rawFeatureEdges[base + 1];
                if (projZ[v1] > -0.5 || projZ[v2] > -0.5) {{
                    ctx3D.moveTo(projX[v1], projY[v1]);
                    ctx3D.lineTo(projX[v2], projY[v2]);
                }}
            }}
            ctx3D.stroke();
        }}

        // Dominant Contact Shadow Sync
        const shadow = document.getElementById("shadowDisc");
        if (shadow) {{
            const sScale = 0.90 + pulseScale * 0.10;
            shadow.style.transform = `scale(${{sScale}})`;
            shadow.style.opacity = `${{0.80 + (pulseScale - 1.0) * 1.6}}`;
        }}

        requestAnimationFrame(render3D);
    }}

    // =========================================================================
    // 2D VECTOR OSCILLOSCOPE
    // =========================================================================
    const scopeCanvas = document.getElementById("scopeCanvas");
    const scopeCtx = scopeCanvas.getContext("2d");

    function renderOscilloscope() {{
        const rect = scopeCanvas.getBoundingClientRect();
        const w = rect.width, h = rect.height;

        if (scopeCanvas.width !== w * window.devicePixelRatio) {{
            scopeCanvas.width = w * window.devicePixelRatio;
            scopeCanvas.height = h * window.devicePixelRatio;
            scopeCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
        }}

        scopeCtx.clearRect(0, 0, w, h);

        // Clinical Grid
        scopeCtx.strokeStyle = "#ECECF0";
        scopeCtx.lineWidth = 1;
        const stepX = w / 15, stepY = h / 6;

        scopeCtx.beginPath();
        for (let x = 0; x <= w; x += stepX) {{
            scopeCtx.moveTo(x, 0); scopeCtx.lineTo(x, h);
        }}
        for (let y = 0; y <= h; y += stepY) {{
            scopeCtx.moveTo(0, y); scopeCtx.lineTo(w, y);
        }}
        scopeCtx.stroke();

        const padX = 16, padY = 20;
        const drawW = w - padX * 2, drawH = h - padY * 2;

        let maxV = 0, maxI = 0;
        for (let i = 0; i < 187; i++) {{
            if (currentSignal[i] > maxV) {{
                maxV = currentSignal[i]; maxI = i;
            }}
        }}

        const pal = CLASS_PALETTES[activeClass];

        scopeCtx.beginPath();
        for (let i = 0; i < 187; i++) {{
            const px = padX + (i / 186) * drawW;
            const py = (h - padY) - (currentSignal[i] / 1.1) * drawH;
            if (i === 0) scopeCtx.moveTo(px, py);
            else scopeCtx.lineTo(px, py);
        }}
        scopeCtx.lineTo(padX + drawW, h - padY);
        scopeCtx.lineTo(padX, h - padY);
        scopeCtx.closePath();
        scopeCtx.fillStyle = `rgba(${{pal.rgb}}, 0.05)`;
        scopeCtx.fill();

        scopeCtx.beginPath();
        for (let i = 0; i < 187; i++) {{
            const px = padX + (i / 186) * drawW;
            const py = (h - padY) - (currentSignal[i] / 1.1) * drawH;
            if (i === 0) scopeCtx.moveTo(px, py);
            else scopeCtx.lineTo(px, py);
        }}
        scopeCtx.strokeStyle = pal.color;
        scopeCtx.lineWidth = 2.0;
        scopeCtx.stroke();

        const rPx = padX + (maxI / 186) * drawW;
        const rPy = (h - padY) - (maxV / 1.1) * drawH;
        scopeCtx.fillStyle = pal.color;
        scopeCtx.beginPath();
        scopeCtx.arc(rPx, rPy, 4.0, 0, Math.PI * 2);
        scopeCtx.fill();

        document.getElementById("peakVal").textContent = `${{maxV.toFixed(3)}} V`;
        document.getElementById("peakVal").style.color = pal.color;
    }}

    window.onload = () => {{
        selectPresetClass(0);
        requestAnimationFrame(render3D);
    }};
</script>

</body>
</html>
"""

components.html(app_html, height=730, scrolling=False)
