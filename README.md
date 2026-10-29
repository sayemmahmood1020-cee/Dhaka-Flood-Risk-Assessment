
# Probabilistic Urban Flood Risk & Structural Damage Assessment Framework

This repository provides a modular, city-agnostic Python codebase for probabilistic flood hazard modeling, building-level structural damage estimation, and spatial historical validation. 

The framework is configured out-of-the-box for **Dhaka, Bangladesh** using MERIT DEM and Google Open Buildings data, but the core architecture is **fully transferable to any city or region**. Users only need to plug in their local Digital Elevation Model (DEM) and building footprint shapefiles.

---

## 📌 Framework Overview

The methodology is split into three core production scripts designed to be executed sequentially:

### 1. `01_flood_monte_carlo_dhaka_v2.py` — Probabilistic Hazard Simulation
- **Monte Carlo Engine**: Runs 5,000 spatial simulations per return period scenario (Frequent: 5-yr, Moderate: 20-yr, Extreme: 100-yr) to propagate DEM vertical error (RMSE = 2.0m) using spatial autocorrelation.
- **Absolute WSE Inundation**: Computes spatial depth based on regional base river stage (auto-detected 5th percentile) plus sampled flood rise. Enforces a maximum physical urban depth ceiling of 5.0m.
- **Vectorized Zonal Statistics**: Uses fast rasterization (`scipy.ndimage.mean`) to extract building-level water depth samples across all realizations. Prefers pre-computed Google Open Buildings areas (`area_in_me`) to prevent projection errors.
- **Outputs**: Probability of Inundation (POI) rasters (0.05m, 0.30m, 1.00m thresholds), 90th percentile depth maps, and a fully propagated uncertainty matrix (`building_depth_samples.npy`).

### 2. `02_add_structural_damage_v2.py` — Risk & Loss Quantification
- **Typology Classification**: Assigns structural typologies (Residential, Commercial, Industrial, Institutional) and applies category-specific depth-damage vulnerability curves.
- **Dynamic Imputation**: Automatically handles zero-area/corrupted geometries by imputing typology-specific median sizes (e.g., 120 sq.m for residential, 280 sq.m for commercial). Correctly reprojects to EPSG:32646 (UTM 46N) for geometric area extraction if needed.
- **EAD & Sensitivity Analysis**: Computes citywide Expected Annual Damage (EAD) using non-parametric bootstrapping (2,000 iterations) across the return period loss distributions. Runs multi-axis sensitivity testing for replacement cost shifts (±20%) and informal settlement curve adaptations.

### 3. `03_historical_validation_template.py` — EO-Based Spatial Validation
- **Historical Benchmarking**: Evaluates modeled extents against Earth Observation flood event maps (ESA EO4SD 2007, 2014, 2016 historical floods).
- **Permanent Water Body Masking**: Automatically isolates river channels and permanent water bodies from the valid evaluation domain to avoid counting them as False Positives.
- **Spatial Contingency Metrics**: Computes Probability of Detection (POD / Hit Rate), False Alarm Rate (FAR), Critical Success Index (CSI / Threat Score), and overall Accuracy across multiple depth and probability thresholds.

---

## 🛠️ Data Requirements for Custom Cities

To adapt this pipeline to another study area, prepare the following inputs and update the paths in `01_flood_monte_carlo_dhaka_v2.py`:

1. **Terrain DEM (`.tif`)**: High-resolution DEM or global products (e.g., MERIT DEM). Set your local `dem_error` (RMSE) configuration.
2. **Building Footprints (`.shp`)**: Polygon features of building boundaries.
3. **Study Area Boundary (`.shp`)**: Vector bounding polygon of the city or watershed domain.
4. *(Optional)* **Historical Flood Extents (`.shp`)**: Satellite-derived flood polygons containing `WATERTYPE` or `WATERCODE` attributes for the validation script.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have a Python environment equipped with standard geospatial data science libraries. Install them via pip:

pip install numpy pandas scipy geopandas rasterio


### Execution Order

Execute the scripts in order:

# Step 1: Run Monte Carlo flood hazard simulation
python 01_flood_monte_carlo_dhaka_v2.py

# Step 2: Compute building-level structural damage and EAD
python 02_add_structural_damage_v2.py

# Step 3: Run spatial validation against historical flood maps
python 03_historical_validation_template.py


---

## 📄 Citation & License

If you use or adapt this codebase in your research, please reference the associated accepted manuscript (information will be updated once the paper is fully published).

> **Sayem Mahmood Sagar**, *"Probabilistic Urban Flood Risk and Building-Level Structural Vulnerability Assessment"*, Accepted for Publication / In Press.

Distributed under the **MIT License**.



