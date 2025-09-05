# PATHOS
Pathology Attention framework for Treatment and Histopathological Outcome Stratification

# Workflow Overview

This repository describes the pipeline for generating features from Whole-Slide Images (WSIs) using **BufferMIL** (attention maps), **cellseg_models** (panoptic segmentation), and a downstream analysis script.

---

## Step 1 – Generating Attention Maps

**Input:**
- Whole-Slide Images (WSIs)  
- Code from https://github.com/FrancescaMiccolis/mil4wsi.git using bufferMIL model

**Processing:**
- Run BufferMIL to compute attention scores over WSI tiles.

**Output:**
- JSON attention maps (heatmaps showing attended regions)

---

## Step 2 – Panoptic Segmentation

**Input:**
- Raw WSIs

**Processing:**
- Run panoptic segmentation with https://github.com/okunator/cellseg_models.pytorch.git (v0.1.21 employed in our analysis)
- Merge segmented areas and cells

**Output:**
- Segmentation **GeoJSON** with cell- and tissue-level annotations

This step is independent of Step 1 and can be run in parallel.

---

## Step 3 – Downstream Analysis

**Input:**
- Attention maps JSON (from Step 1)  
- Panoptic segmentation GeoJSON (from Step 2)  
- Analysis script provided in this repository

  **Processing:**
- Integrates attention regions (BufferMIL) with segmentation results (cellseg_models).
- The only parameter to set is the **attention patch level to discard** (default = `5`).  
  → Keeps patches with levels `1–4` and filters out the least attended patches.
- Computes a panel of morphological and spatial features.

**Output:**
Numerical feature set, we focused on 7 key features that, according to our analysis, were the most informative among all:

1. `tumor_cells_eccentricity_max`  
2. `tumor_cells_major_axis_len_max`  
3. `tumor_cells_fractal_dimension_mean`  
4. `stromal_n_cells`  
5. `distal_stroma_area`  
6. `stroma_area`  
7. `stromal_cell_proportion`
