import os
import gc
import json
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import rasterize

# ----------------- PATH CONFIGURATION -----------------
PROJECT_DIR = r'E:\Download\Academic\3-1\Article\FLOOD_STRUCT\Flood'
HISTORIC_DIR = r'E:\Download\Academic\3-1\Article\Historic'
OUTPUT_FOLDER = os.path.join(PROJECT_DIR, 'flood_results_q1_v2')

HISTORICAL_EVENTS = {
    '2007': {'scenario': 'moderate', 'description': 'ESA EO4SD 2007 Dhaka Severe Flood'},
    '2014': {'scenario': 'frequent', 'description': 'ESA EO4SD 2014 Dhaka Moderate Flood'},
    '2016': {'scenario': 'frequent', 'description': 'ESA EO4SD 2016 Dhaka Moderate Flood'}
}

# ----------------- UTILITY & LOADERS -----------------

def find_shapefile_by_year(search_dir, year):
    """Recursively finds .shp file containing target year."""
    for root, _, files in os.walk(search_dir):
        for file in files:
            if file.lower().endswith('.shp') and year in file:
                return os.path.join(root, file)
    return None


def load_modeled_extent(scenario_name, method='poi', depth_m=0.3, prob_threshold=0.5):
    """
    Loads modeled flood extent based on POI rasters or mean depth rasters.
    Includes explicit debugging to verify path loading and array values.
    """
    scenario_dir = os.path.join(OUTPUT_FOLDER, scenario_name)
    
    if method == 'poi':
        # Candidate POI filenames
        d_str = f"{depth_m:.2f}".rstrip('0').rstrip('.')  # e.g., '0.3' or '1'
        d_str2 = f"{depth_m:.2f}"                          # e.g., '0.30' or '1.00'
        d_str3 = f"{depth_m:.1f}"                          # e.g., '0.3' or '1.0'
        
        possible_names = list(dict.fromkeys([
            f'poi_{d_str}m.tif',
            f'poi_{d_str2}m.tif',
            f'poi_{d_str3}m.tif',
            f'poi_{depth_m}m.tif'
        ]))
        
        raster_path = None
        for name in possible_names:
            p = os.path.join(scenario_dir, name)
            if os.path.exists(p):
                raster_path = p
                break
                
        if not raster_path:
            raise FileNotFoundError(
                f"POI raster for depth {depth_m}m not found in {scenario_dir}. Looked for: {possible_names}"
            )
            
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            profile = src.profile.copy()
            nodata = src.nodata if src.nodata is not None else -9999.0
            
            valid_mask = (data != nodata) & (~np.isnan(data))
            
            # Detect percentage vs probability scale
            data_max = float(np.nanmax(data[valid_mask])) if np.any(valid_mask) else 0.0
            if data_max > 1.0:
                p_thresh = prob_threshold * 100.0
            else:
                p_thresh = prob_threshold
                
            modeled_mask = valid_mask & (data >= p_thresh)
            
            print(f"    [FILE LOADED] {os.path.basename(raster_path)} | Max Value: {data_max:.2f} | Threshold Used: {p_thresh} | Flooded Px: {modeled_mask.sum():,}")
            return modeled_mask, valid_mask, profile

    elif method == 'mean_depth':
        raster_path = os.path.join(scenario_dir, 'mean_depth.tif')
        if not os.path.exists(raster_path):
            raise FileNotFoundError(f"Mean depth raster not found: {raster_path}")
            
        with rasterio.open(raster_path) as src:
            depth = src.read(1)
            profile = src.profile.copy()
            nodata = src.nodata if src.nodata is not None else -9999.0
            
            valid_mask = (depth != nodata) & (~np.isnan(depth))
            modeled_mask = valid_mask & (depth >= depth_m)
            
            print(f"    [FILE LOADED] {os.path.basename(raster_path)} | Max Depth: {np.nanmax(depth[valid_mask]):.2f}m | Flooded Px: {modeled_mask.sum():,}")
            return modeled_mask, valid_mask, profile


def rasterize_historical_shapefile(shp_path, ref_profile, base_valid_mask):
    """
    Reads EO4SD shapefile and generates:
      1. observed_flood_mask (Flooded == True)
      2. updated_valid_mask (base_valid_mask minus permanent water bodies)
    """
    gdf = gpd.read_file(shp_path)
    wt_str = gdf['WATERTYPE'].astype(str).str.lower().str.strip() if 'WATERTYPE' in gdf.columns else gpd.GeoSeries()

    # 1. Extract Flooded Polygons
    if 'WATERTYPE' in gdf.columns:
        flood_gdf = gdf[wt_str.str.contains('flood') & (~wt_str.str.contains('non'))]
    elif 'WATERCODE' in gdf.columns:
        flood_gdf = gdf[(gdf['WATERCODE'] == 1.0) | (gdf['WATERCODE'] == 1)]
    elif 'DN' in gdf.columns:
        flood_gdf = gdf[gdf['DN'] == 1]
    else:
        flood_gdf = gdf.copy()

    # 2. Extract Permanent Water Body Polygons
    if 'WATERTYPE' in gdf.columns:
        perm_water_gdf = gdf[wt_str.str.contains('water body') | wt_str.str.contains('waterbody')]
    elif 'WATERCODE' in gdf.columns:
        perm_water_gdf = gdf[gdf['WATERCODE'] == 3.0]
    else:
        perm_water_gdf = gpd.GeoDataFrame()

    height, width = ref_profile['height'], ref_profile['width']
    ref_transform = ref_profile['transform']

    # Rasterize Flooded Polygons
    if len(flood_gdf) > 0:
        flood_gdf = flood_gdf.to_crs(epsg=4326)
        flood_gdf['geometry'] = flood_gdf.geometry.buffer(0)
        flood_shapes = [(geom, 1) for geom in flood_gdf.geometry if not geom.is_empty]
        obs_raster = rasterize(flood_shapes, out_shape=(height, width), transform=ref_transform, fill=0, default_value=1, dtype=np.uint8)
    else:
        obs_raster = np.zeros((height, width), dtype=np.uint8)

    # Rasterize Permanent Water Bodies
    if len(perm_water_gdf) > 0:
        perm_water_gdf = perm_water_gdf.to_crs(epsg=4326)
        perm_water_gdf['geometry'] = perm_water_gdf.geometry.buffer(0)
        water_shapes = [(geom, 1) for geom in perm_water_gdf.geometry if not geom.is_empty]
        water_raster = rasterize(water_shapes, out_shape=(height, width), transform=ref_transform, fill=0, default_value=1, dtype=np.uint8)
    else:
        water_raster = np.zeros((height, width), dtype=np.uint8)

    # Exclude Permanent Water Bodies from the valid evaluation domain
    perm_water_mask = (water_raster == 1) & base_valid_mask
    updated_valid_mask = base_valid_mask & (~perm_water_mask)
    observed_flood_mask = (obs_raster == 1) & updated_valid_mask

    base_val_px = int(base_valid_mask.sum())
    perm_water_px = int(perm_water_mask.sum())
    net_val_px = int(updated_valid_mask.sum())
    pct_excluded = (perm_water_px / base_val_px * 100) if base_val_px > 0 else 0.0

    print(f"  [DOMAIN MASKING] Raw Valid Px: {base_val_px:,} | Perm Water Excluded: {perm_water_px:,} ({pct_excluded:.2f}%) | Net Valid Px: {net_val_px:,}")

    return observed_flood_mask, updated_valid_mask


def compute_metrics(modeled_mask, observed_mask, valid_mask):
    """Calculates spatial metrics cleanly using explicit array masks."""
    mod = modeled_mask & valid_mask
    obs = observed_mask & valid_mask

    tp_mask = mod & obs
    fp_mask = mod & (~obs)
    fn_mask = (~mod) & obs
    tn_mask = (~mod) & (~obs) & valid_mask

    tp = int(tp_mask.sum())
    fp = int(fp_mask.sum())
    fn = int(fn_mask.sum())
    tn = int(tn_mask.sum())

    total_valid = int(valid_mask.sum())

    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    acc = (tp + tn) / total_valid if total_valid > 0 else 0.0
    mod_cov = ((tp + fp) / total_valid * 100) if total_valid > 0 else 0.0

    return {
        'POD': round(pod * 100, 2),
        'FAR': round(far * 100, 2),
        'CSI': round(csi * 100, 2),
        'Accuracy': round(acc * 100, 2),
        'Modeled_Coverage_Pct': round(mod_cov, 2),
        'Hits_TP': tp,
        'FalseAlarms_FP': fp,
        'Misses_FN': fn,
        'TrueNeg_TN': tn
    }

# ----------------- MAIN EXECUTION LOOP -----------------
if __name__ == '__main__':
    print("=" * 95)
    print("      MULTI-THRESHOLD HISTORICAL VALIDATION (WATER-MASKED & FIXED)")
    print("=" * 95)

    TEST_CONFIGS = [
        {'name': 'POI >= 0.30m (P >= 0.50)', 'method': 'poi', 'depth': 0.3, 'prob': 0.5},
        {'name': 'POI >= 1.00m (P >= 0.50)', 'method': 'poi', 'depth': 1.0, 'prob': 0.5},
        {'name': 'Mean Depth >= 0.30m', 'method': 'mean_depth', 'depth': 0.3, 'prob': None},
        {'name': 'Mean Depth >= 1.00m', 'method': 'mean_depth', 'depth': 1.0, 'prob': None},
    ]

    summary_table = []

    for year, info in HISTORICAL_EVENTS.items():
        shp_path = find_shapefile_by_year(HISTORIC_DIR, year)
        if not shp_path:
            print(f"\nSkipping {year}: shapefile not found.")
            continue

        print(f"\n>>> Evaluating Event Year: {year} (Scenario: {info['scenario']}) <<<")
        print(f"  Shapefile: {os.path.basename(shp_path)}")

        for cfg in TEST_CONFIGS:
            try:
                # 1. Load Modeled Raster Extent
                modeled_mask, base_valid_mask, profile = load_modeled_extent(
                    scenario_name=info['scenario'],
                    method=cfg['method'],
                    depth_m=cfg['depth'],
                    prob_threshold=cfg['prob']
                )

                # 2. Rasterize EO4SD Shapefile & Mask Permanent Water Bodies
                observed_mask, updated_valid_mask = rasterize_historical_shapefile(
                    shp_path, profile, base_valid_mask
                )

                # 3. Compute Metrics on Cleaned Domain
                metrics = compute_metrics(modeled_mask, observed_mask, updated_valid_mask)

                summary_table.append({
                    'Year': year,
                    'Threshold Method': cfg['name'],
                    'Modeled Coverage': f"{metrics['Modeled_Coverage_Pct']}%",
                    'POD (Hit Rate)': f"{metrics['POD']}%",
                    'FAR (False Alarm)': f"{metrics['FAR']}%",
                    'CSI (Threat)': f"{metrics['CSI']}%",
                    'Accuracy': f"{metrics['Accuracy']}%"
                })

                print(f"    --> Results: Modeled Cov: {metrics['Modeled_Coverage_Pct']}% | POD: {metrics['POD']}% | FAR: {metrics['FAR']}% | CSI: {metrics['CSI']}% | Acc: {metrics['Accuracy']}%")

            except Exception as e:
                print(f"    [ERROR] {cfg['name']}: {e}")

        gc.collect()

    print("\n" + "=" * 95)
    print("      FINAL MULTI-THRESHOLD SENSITIVITY SUMMARY MATRIX")
    print("=" * 95)
    
    print(f"| {'Year':<6} | {'Threshold Method':<25} | {'Modeled Cov %':<15} | {'POD %':<8} | {'FAR %':<8} | {'CSI %':<8} | {'Accuracy %':<10} |")
    print("|" + "-"*8 + "|" + "-"*27 + "|" + "-"*17 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*12 + "|")
    for row in summary_table:
        print(f"| {row['Year']:<6} | {row['Threshold Method']:<25} | {row['Modeled Coverage']:<15} | {row['POD (Hit Rate)']:<8} | {row['FAR (False Alarm)']:<8} | {row['CSI (Threat)']:<8} | {row['Accuracy']:<10} |")
    print("=" * 95)
