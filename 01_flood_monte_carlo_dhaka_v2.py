import os
import gc
import json
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.mask import mask
from rasterio.features import rasterize
from scipy.ndimage import gaussian_filter
from scipy import ndimage

# ----------------- REPRODUCIBILITY -----------------
np.random.seed(42)

# ----------------- PATHS (Colab-aware) -----------------
try:
    IN_COLAB = 'google.colab' in str(get_ipython())
except NameError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    base_path = 'file path for used google colab folder'
else:
    base_path = r'base file path'

dem_path = os.path.join(base_path, 'dem file')
study_area_path = os.path.join(base_path, 'shape file')
buildings_path = os.path.join(base_path, 'buildings shape file')
output_folder = os.path.join(base_path, 'output file')
checkpoint_folder = os.path.join(output_folder, 'checkpoints')
os.makedirs(output_folder, exist_ok=True)
os.makedirs(checkpoint_folder, exist_ok=True)

# ----------------- CONFIGURATION -----------------
# DEM vertical error (RMSE in meters)
dem_error = 2.0

# Physical upper bound for overland urban inundation in Dhaka (meters)
MAX_PHYSICAL_DEPTH = 5.0

n_simulations = 5000
NODATA_VALUE = -9999.0

scenarios = {
    'frequent': {'mean_rise': 1.0, 'std_rise': 0.3, 'return_period': 5},
    'moderate': {'mean_rise': 2.0, 'std_rise': 0.5, 'return_period': 20},
    'extreme':  {'mean_rise': 3.0, 'std_rise': 0.8, 'return_period': 100},
}

# Spatial autocorrelation radius for DEM error field
CORRELATION_RADIUS = 5.0  # pixels (~150 m at 30 m resolution)

# Probability-of-inundation thresholds
POI_THRESHOLDS = [0.05, 0.30, 1.00]

# Uncertainty-propagation controls
SUBSAMPLE_INTERVAL = 5
RASTER_SNAPSHOT_INTERVAL = 50
CHECKPOINT_EVERY = 500

# ----------------- LOAD & CLIP DEM -----------------
print("Loading and clipping MERIT DEM to study area...")
study_gdf = gpd.read_file(study_area_path)
study_geom = study_gdf.union_all()
with rasterio.open(dem_path) as src:
    clipped_array, clipped_transform = mask(src, [study_geom], crop=True, filled=False)
    dem = clipped_array[0].astype(np.float32)
    dem = np.ma.filled(dem, np.nan)

    profile = src.profile.copy()
    profile.update({
        'height': dem.shape[0],
        'width': dem.shape[1],
        'transform': clipped_transform,
        'nodata': NODATA_VALUE,
        'dtype': rasterio.float32
    })
    height, width = dem.shape
    raster_crs = src.crs

out_profile = profile.copy()

# Auto-detect baseline channel/river bed elevation (5th percentile DEM)
valid_dem_cells = dem[~np.isnan(dem)]
BASE_RIVER_STAGE = float(np.percentile(valid_dem_cells, 5))
print(f"Base Channel/River Stage auto-detected: {BASE_RIVER_STAGE:.2f} m PWD")

# ----------------- UNIFIED BUILDING-LEVEL ZONAL STATS -----------------
print("Rasterizing building footprints for vectorized zonal stats...")
buildings_gdf = gpd.read_file(buildings_path).to_crs(raster_crs)
buildings_gdf = buildings_gdf.reset_index(drop=True)
buildings_gdf['bld_id'] = np.arange(len(buildings_gdf))

shapes = list(zip(buildings_gdf.geometry, buildings_gdf['bld_id']))
building_id_raster = rasterize(
    shapes, out_shape=(height, width), transform=clipped_transform,
    fill=-1, all_touched=True, dtype=np.int32
)

valid_bld_ids = np.unique(building_id_raster[building_id_raster >= 0])
n_buildings = len(valid_bld_ids)
print(f"  {n_buildings} of {len(buildings_gdf)} buildings received >=1 raster cell.")

buildings_gdf[['bld_id']].to_csv(os.path.join(output_folder, 'building_id_index.csv'), index=False)

# Compute footprint area
print("Parsing building footprint areas...")
if 'area_in_me' in buildings_gdf.columns:
    print("  Using pre-computed 'area_in_me' attribute from Google Open Buildings dataset.")
    buildings_gdf['area_m2'] = buildings_gdf['area_in_me'].astype(float)
else:
    print("  Projecting to EPSG:32646 to calculate geometric footprint area...")
    buildings_utm = buildings_gdf.to_crs(epsg=32646)
    buildings_utm.geometry = buildings_utm.geometry.buffer(0)
    buildings_gdf['area_m2'] = buildings_utm.geometry.area

zero_area = (buildings_gdf['area_m2'] <= 0).sum()
if zero_area > 0:
    print(f"  {zero_area} zero/invalid area buildings -> replaced with 80.0 m^2 fallback.")
    buildings_gdf['area_m2'] = buildings_gdf['area_m2'].replace(0, 80.0)

# Export building attributes
buildings_gdf_4326 = buildings_gdf.to_crs(epsg=4326)
buildings_gdf_4326.to_file(
    os.path.join(output_folder, 'building_attributes.gpkg'), driver='GPKG'
)
buildings_gdf.drop(columns='geometry').to_csv(
    os.path.join(output_folder, 'building_attributes.csv'), index=False
)

# ----------------- MONTE CARLO FOR EACH SCENARIO -----------------
for scenario_name, scenario_params in scenarios.items():
    print(f"\nRunning scenario: {scenario_name} (mean rise={scenario_params['mean_rise']:.1f} m, "
          f"return period={scenario_params['return_period']} yr)")

    scenario_folder = os.path.join(output_folder, scenario_name)
    os.makedirs(scenario_folder, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_folder, f'{scenario_name}_ckpt.npz')

    expected_depth_sum = np.zeros((height, width), dtype=np.float64)
    inundation_counts = {th: np.zeros((height, width), dtype=np.int32) for th in POI_THRESHOLDS}
    raster_snapshots = []

    n_saved = n_simulations // SUBSAMPLE_INTERVAL
    building_depth_samples = np.zeros((n_saved, n_buildings), dtype=np.float32)
    saved_idx = 0

    start_sim = 0
    if os.path.exists(ckpt_path):
        print(f"  Resuming from checkpoint: {ckpt_path}")
        ck = np.load(ckpt_path)
        start_sim = int(ck['i']) + 1
        expected_depth_sum = ck['expected_depth_sum']
        for th in POI_THRESHOLDS:
            inundation_counts[th] = ck[f'poi_{th}']
        building_depth_samples = ck['building_depth_samples']
        saved_idx = int(ck['saved_idx'])
        raster_snapshots = list(ck['raster_snapshots']) if 'raster_snapshots' in ck else []

    for i in range(start_sim, n_simulations):
        if i % 1000 == 0:
            print(f"  Simulation {i}/{n_simulations}")

        # Spatially correlated DEM vertical error
        uncorrelated_error = np.random.normal(0, dem_error, dem.shape)
        correlated_error = gaussian_filter(uncorrelated_error, sigma=CORRELATION_RADIUS)
        correlated_error = np.clip(correlated_error, -2 * dem_error, 2 * dem_error)
        perturbed = dem + correlated_error

        # Truncated normal flood rise
        sampled_flood_rise = max(0.0, np.random.normal(
            scenario_params['mean_rise'], scenario_params['std_rise']))

        # ABSOLUTE WATER SURFACE ELEVATION (WSE)
        # Regional water stage = Base river stage + sampled flood rise
        wse = BASE_RIVER_STAGE + sampled_flood_rise

        # Depth calculation: water only accumulates where terrain elevation < WSE
        depth = np.maximum(0.0, wse - perturbed)
        depth = np.clip(depth, 0.0, MAX_PHYSICAL_DEPTH)

        valid_mask = ~np.isnan(dem)
        depth[~valid_mask] = np.nan

        expected_depth_sum[valid_mask] += depth[valid_mask]
        for th in POI_THRESHOLDS:
            inundation_counts[th][(depth >= th) & valid_mask] += 1

        # Fast vectorized footprint-average per building
        if i % SUBSAMPLE_INTERVAL == 0 and saved_idx < n_saved:
            depth_filled = np.nan_to_num(depth, nan=0.0)
            bld_means = ndimage.mean(depth_filled, labels=building_id_raster, index=valid_bld_ids)
            building_depth_samples[saved_idx, :] = bld_means
            saved_idx += 1

        if i % RASTER_SNAPSHOT_INTERVAL == 0:
            raster_snapshots.append(np.nan_to_num(depth, nan=0.0).astype(np.float32))

        if (i + 1) % CHECKPOINT_EVERY == 0:
            np.savez_compressed(
                ckpt_path, i=i, expected_depth_sum=expected_depth_sum,
                building_depth_samples=building_depth_samples, saved_idx=saved_idx,
                raster_snapshots=np.array(raster_snapshots, dtype=object),
                **{f'poi_{th}': inundation_counts[th] for th in POI_THRESHOLDS}
            )

        del correlated_error, perturbed, uncorrelated_error
        gc.collect()

    # ----------------- Final raster outputs -----------------
    final_mean_depth = expected_depth_sum / n_simulations
    final_mean_depth[np.isnan(dem)] = NODATA_VALUE

    final_poi = {}
    for th in POI_THRESHOLDS:
        final_poi[th] = inundation_counts[th].astype(np.float32) / n_simulations
        final_poi[th][np.isnan(dem)] = NODATA_VALUE

    if raster_snapshots:
        stack = np.stack(raster_snapshots, axis=0)
        final_p90_depth = np.percentile(stack, 90, axis=0).astype(np.float32)
        final_p90_depth[np.isnan(dem)] = NODATA_VALUE
        del stack
    else:
        final_p90_depth = np.full((height, width), NODATA_VALUE, dtype=np.float32)

    for name, array in {'mean_depth': final_mean_depth, 'p90_depth': final_p90_depth}.items():
        with rasterio.open(os.path.join(scenario_folder, f'{name}.tif'), 'w', **out_profile) as dst:
            dst.write(np.nan_to_num(array, nan=NODATA_VALUE), 1)

    for th in POI_THRESHOLDS:
        with rasterio.open(os.path.join(scenario_folder, f'poi_{th}m.tif'), 'w', **out_profile) as dst:
            dst.write(np.nan_to_num(final_poi[th], nan=NODATA_VALUE), 1)

    # ----------------- Building-level outputs -----------------
    np.save(os.path.join(scenario_folder, 'building_depth_samples.npy'), building_depth_samples)
    np.save(os.path.join(scenario_folder, 'building_ids.npy'), valid_bld_ids)

    summary_mean = building_depth_samples.mean(axis=0)
    summary_p5 = np.percentile(building_depth_samples, 5, axis=0)
    summary_p95 = np.percentile(building_depth_samples, 95, axis=0)
    np.savez(os.path.join(scenario_folder, 'building_depth_summary.npz'),
             bld_id=valid_bld_ids, mean=summary_mean, p5=summary_p5, p95=summary_p95)

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

# ----------------- Run manifest -----------------
manifest = {
    'n_simulations': n_simulations,
    'dem_error_rmse_m': dem_error,
    'max_physical_depth_m': MAX_PHYSICAL_DEPTH,
    'correlation_radius_px': CORRELATION_RADIUS,
    'poi_thresholds_m': POI_THRESHOLDS,
    'subsample_interval': SUBSAMPLE_INTERVAL,
    'n_saved_samples_per_scenario': n_simulations // SUBSAMPLE_INTERVAL,
    'scenarios': scenarios,
    'n_buildings_matched': int(n_buildings),
    'base_river_stage_m_pwd': BASE_RIVER_STAGE,
    'random_seed': 42,
}
with open(os.path.join(output_folder, 'run_manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2)

print("\n" + "=" * 80)
print("Monte Carlo flood simulation complete.")
print(f"Outputs saved to: {output_folder}")
print("=" * 80)
