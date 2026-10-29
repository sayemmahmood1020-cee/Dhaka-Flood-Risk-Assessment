import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd

# Support compatibility across NumPy 1.x and 2.x versions
_integrate_trapezoid = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')

# -------------------------------------------------------------------------
# 1. CONFIGURATION & DIRECTORY PATHS
# -------------------------------------------------------------------------
try:
    IN_COLAB = 'google.colab' in str(get_ipython())
except NameError:
    IN_COLAB = False

if IN_COLAB:
    BASE_DIR = 'base direction'
else:
    BASE_DIR = r'base direction'

SCENARIOS = ['frequent', 'moderate', 'extreme']
ANNUAL_EXCEEDANCE_PROB = {'frequent': 0.20, 'moderate': 0.05, 'extreme': 0.01}

# Physical clamping threshold for ground-level building inundation in Dhaka
MAX_PHYSICAL_DEPTH_M = 5.0  

TAIL_ASSUMPTION = 'truncate'
BOOTSTRAP_ITERATIONS = 2000

# -------------------------------------------------------------------------
# 2. BUILDING CLASSIFICATION & TYPOLOGY MEDIANS
# -------------------------------------------------------------------------
TYPOLOGY_MEDIAN_AREAS = {
    'residential': 120.0,
    'commercial': 280.0,
    'industrial': 450.0,
    'institutional': 350.0
}

def classify_buildings(gdf):
    """
    Assigns a statistically representative structural typology mix to Dhaka's
    building inventory.
    """
    np.random.seed(42)
    
    typologies = ['residential', 'commercial', 'industrial', 'institutional']
    probabilities = [0.70, 0.15, 0.10, 0.05]
    
    total_buildings = len(gdf)
    random_assignments = np.random.choice(typologies, size=total_buildings, p=probabilities)
    
    return pd.Series(random_assignments, index=gdf.index).astype(str)

# -------------------------------------------------------------------------
# 3. DEPTH-DAMAGE CURVES & VALUATION INVENTORY
# -------------------------------------------------------------------------
CURVES_STANDARD = {
    'residential':   {'thresholds': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0], 'ratios': [0.0, 0.15, 0.35, 0.55, 0.70, 0.85]},
    'commercial':    {'thresholds': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0], 'ratios': [0.0, 0.20, 0.45, 0.65, 0.80, 0.90]},
    'industrial':    {'thresholds': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0], 'ratios': [0.0, 0.18, 0.40, 0.60, 0.75, 0.88]},
    'institutional': {'thresholds': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0], 'ratios': [0.0, 0.18, 0.42, 0.62, 0.78, 0.90]},
}

INFORMAL_SHIFT_FACTOR = 1.25

def generate_informal_curves(standard_curves, scale_factor):
    informal_curves = {}
    for typology, curve in standard_curves.items():
        scaled_ratios = np.clip(np.array(curve['ratios']) * scale_factor, 0.0, 1.0)
        informal_curves[typology] = {
            'thresholds': curve['thresholds'],
            'ratios': list(scaled_ratios)
        }
    return informal_curves

CURVES_INFORMAL = generate_informal_curves(CURVES_STANDARD, INFORMAL_SHIFT_FACTOR)

REPLACEMENT_COST_M2 = {
    'residential': 200, 'commercial': 350, 'industrial': 300, 'institutional': 320,
}

# -------------------------------------------------------------------------
# 4. LOSS CALCULATION ENGINE
# -------------------------------------------------------------------------
def compute_scenario_loss_distribution(depth_samples, building_types, areas, curve_set):
    loss_matrix = np.zeros_like(depth_samples, dtype=np.float64)

    for typology, curve in curve_set.items():
        type_mask = (building_types == typology)
        if not type_mask.any():
            continue
            
        depths_subset = depth_samples[:, type_mask]
        interpolated_ratios = np.interp(
            depths_subset.ravel(), curve['thresholds'], curve['ratios']
        ).reshape(depths_subset.shape)
        
        asset_values = areas[type_mask] * REPLACEMENT_COST_M2[typology]
        loss_matrix[:, type_mask] = interpolated_ratios * asset_values[np.newaxis, :]

    total_losses_per_draw = loss_matrix.sum(axis=1)
    average_loss_per_asset = loss_matrix.mean(axis=0)
    
    return total_losses_per_draw, average_loss_per_asset

def bootstrap_expected_annual_damage(scenario_losses, probabilities, tail_assumption='truncate', n_bootstrap=2000):
    sorted_scenarios = sorted(probabilities.keys(), key=lambda s: probabilities[s], reverse=True)
    prob_vectors = np.array([probabilities[s] for s in sorted_scenarios])

    ead_distribution = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sampled_losses = np.array([np.random.choice(scenario_losses[s]) for s in sorted_scenarios])

        if tail_assumption == 'truncate':
            aep_axis = np.concatenate([[1.0], prob_vectors])
            loss_axis = np.concatenate([[0.0], sampled_losses])
        elif tail_assumption == 'extrapolate':
            slope = (sampled_losses[-1] - sampled_losses[-2]) / (prob_vectors[-1] - prob_vectors[-2])
            extrap_prob = 0.002
            extrap_loss = max(sampled_losses[-1] + slope * (extrap_prob - prob_vectors[-1]), sampled_losses[-1])
            aep_axis = np.concatenate([[1.0], prob_vectors, [extrap_prob]])
            loss_axis = np.concatenate([[0.0], sampled_losses, [extrap_loss]])
        else:
            raise ValueError("Invalid tail_assumption configured.")

        ascending_order = np.argsort(aep_axis)
        ead_distribution[i] = _integrate_trapezoid(loss_axis[ascending_order], aep_axis[ascending_order])

    return ead_distribution

# -------------------------------------------------------------------------
# 5. MAIN EXECUTION PIPELINE
# -------------------------------------------------------------------------
if __name__ == '__main__':
    cached_typologies = {}
    scenario_loss_draws = {}
    asset_summaries = {}

    # 1. Load and reproject building attributes ONCE with explicit CRS handling
    path_attributes = os.path.join(BASE_DIR, 'building_attributes.gpkg')
    print("Loading building attribute layer...")
    attributes_gdf = gpd.read_file(path_attributes)

    # Read the pre-computed real building footprint area directly from Google Open Buildings
    attributes_gdf['area_m2'] = attributes_gdf['area_in_me'].astype(float)
    attributes_gdf = attributes_gdf.set_index('bld_id')

    for scenario in SCENARIOS:
        print(f"\n--- Processing Scenario: {scenario.upper()} ---")
        scenario_path = os.path.join(BASE_DIR, scenario)
        
        path_depths = os.path.join(scenario_path, 'building_depth_samples.npy')
        path_ids = os.path.join(scenario_path, 'building_ids.npy')

        if not (os.path.exists(path_depths) and os.path.exists(path_ids)):
            print(f"  Missing input matrices for {scenario}. Skipping.")
            continue

        depth_matrix = np.load(path_depths)
        building_ids = np.load(path_ids)
        
        # Align building IDs
        valid_records = np.isin(building_ids, attributes_gdf.index)
        depth_matrix = depth_matrix[:, valid_records]
        building_ids = building_ids[valid_records]
        
        scenario_attributes = attributes_gdf.loc[building_ids].copy()

        # Clamp physically unfeasible depths (DEM sinks/pits)
        uncapped_max = depth_matrix.max()
        depth_matrix = np.clip(depth_matrix, 0.0, MAX_PHYSICAL_DEPTH_M)

        # Assign typologies uniformly across scenarios
        if 'building_type' not in cached_typologies:
            simulated_types = classify_buildings(scenario_attributes)
            cached_typologies['building_type'] = simulated_types.values
            
            output_classification = pd.DataFrame({
                'bld_id': building_ids,
                'building_type': simulated_types.values,
                'classification_source': 'probabilistic_profile'
            })
            output_classification.to_csv(os.path.join(BASE_DIR, 'building_classification.csv'), index=False)

        assigned_types = cached_typologies['building_type']
        scenario_attributes['building_type'] = assigned_types

        # Handle zero/corrupted areas via typology-specific medians
        invalid_area_mask = (scenario_attributes['area_m2'] <= 1.0) | (scenario_attributes['area_m2'].isnull())
        if invalid_area_mask.sum() > 0:
            print(f"  Imputing {invalid_area_mask.sum():,} invalid/zero areas using typology medians...")
            for typ, med_area in TYPOLOGY_MEDIAN_AREAS.items():
                fix_mask = invalid_area_mask & (scenario_attributes['building_type'] == typ)
                scenario_attributes.loc[fix_mask, 'area_m2'] = med_area

        calculated_areas = scenario_attributes['area_m2'].values

        # Diagnostic Status
        print(f"  Shape of depth samples matrix : {depth_matrix.shape}")
        print(f"  Raw Max Depth (Pre-clamp)     : {uncapped_max:.2f} m")
        print(f"  Clamped Max Depth             : {depth_matrix.max():.2f} m")
        print(f"  Footprint Area Range (m2)     : Min={calculated_areas.min():.1f}, Median={np.median(calculated_areas):.1f}, Max={calculated_areas.max():.1f}")

        # Compute risk losses
        losses_computed, building_mean_loss = compute_scenario_loss_distribution(
            depth_matrix, assigned_types, calculated_areas, CURVES_STANDARD
        )
        scenario_loss_draws[scenario] = losses_computed

        # Save individual file logs
        run_summary = pd.DataFrame({
            'bld_id': building_ids,
            'building_type': assigned_types,
            'area_m2': calculated_areas,
            'mean_loss_usd': building_mean_loss,
        })
        asset_summaries[scenario] = run_summary
        run_summary.to_csv(os.path.join(scenario_path, 'building_loss_summary.csv'), index=False)

        print(f"  Citywide loss metrics summary:")
        print(f"    Mean:  ${losses_computed.mean()/1e6:,.1f}M")
        print(f"    P5:    ${np.percentile(losses_computed, 5)/1e6:,.1f}M")
        print(f"    P95:   ${np.percentile(losses_computed, 95)/1e6:,.1f}M")

    # -------------------------------------------------------------------------
    # 6. RISK INTEGRATION & PROBABILISTIC EAD BOOTSTRAP
    # -------------------------------------------------------------------------
    if len(scenario_loss_draws) == len(SCENARIOS):
        print("\n--- Citywide EAD (Bootstrap Over Scenario Loss Distributions) ---")
        ead_samples = bootstrap_expected_annual_damage(
            scenario_loss_draws, ANNUAL_EXCEEDANCE_PROB,
            tail_assumption=TAIL_ASSUMPTION, n_bootstrap=BOOTSTRAP_ITERATIONS
        )

        ead_json_output = {
            'mean_usd': float(ead_samples.mean()),
            'median_usd': float(np.percentile(ead_samples, 50)),
            'p5_usd': float(np.percentile(ead_samples, 5)),
            'p95_usd': float(np.percentile(ead_samples, 95)),
            'std_usd': float(ead_samples.std()),
            'tail_assumption': TAIL_ASSUMPTION,
            'n_bootstrap': BOOTSTRAP_ITERATIONS,
        }
        print(json.dumps(ead_json_output, indent=2))

        manuscript_risk_curve = [{'aep': 1.0, 'return_period_yr': 1, 'loss_usd_mean': 0.0, 'loss_usd_p5': 0.0, 'loss_usd_p95': 0.0}]
        for s in SCENARIOS:
            manuscript_risk_curve.append({
                'aep': ANNUAL_EXCEEDANCE_PROB[s],
                'return_period_yr': int(round(1 / ANNUAL_EXCEEDANCE_PROB[s])),
                'loss_usd_mean': float(scenario_loss_draws[s].mean()),
                'loss_usd_p5': float(np.percentile(scenario_loss_draws[s], 5)),
                'loss_usd_p95': float(np.percentile(scenario_loss_draws[s], 95)),
            })
            
        pd.DataFrame(manuscript_risk_curve).to_csv(
            os.path.join(BASE_DIR, 'risk_curve_for_manuscript_figure.csv'), index=False
        )
        with open(os.path.join(BASE_DIR, 'ead_summary.json'), 'w') as f:
            json.dump(ead_json_output, f, indent=2)

        print(f"\nSaved: risk_curve_for_manuscript_figure.csv")
        print(f"Saved: ead_summary.json")

    # -------------------------------------------------------------------------
    # 7. MULTI-AXIS SENSITIVITY TESTING
    # -------------------------------------------------------------------------
    print("\n--- Running Extended Sensitivity Analysis ---")
    sensitivity_records = []
    
    for scenario, standard_draws in scenario_loss_draws.items():
        active_summary = asset_summaries.get(scenario)
        if active_summary is None:
            continue
            
        depth_samples_raw = np.load(os.path.join(BASE_DIR, scenario, 'building_depth_samples.npy'))
        depth_samples_raw = depth_samples_raw[:, valid_records]
        depth_samples_raw = np.clip(depth_samples_raw, 0.0, MAX_PHYSICAL_DEPTH_M)
        
        current_types = cached_typologies['building_type']
        current_areas = active_summary['area_m2'].values

        # Axis A: Valuation Shift Scale Bounds (+/- 20%)
        for scale_multiplier, label_string in [(0.8, 'cost_-20%'), (1.0, 'cost_baseline'), (1.2, 'cost_+20%')]:
            cached_base_costs = REPLACEMENT_COST_M2.copy()
            for key in REPLACEMENT_COST_M2:
                REPLACEMENT_COST_M2[key] = cached_base_costs[key] * scale_multiplier
                
            loss_sensitivity_draws, _ = compute_scenario_loss_distribution(
                depth_samples_raw, current_types, current_areas, CURVES_STANDARD
            )
            sensitivity_records.append({
                'scenario': scenario, 'axis': 'cost', 'variant': label_string,
                'mean_loss_usd': float(loss_sensitivity_draws.mean())
            })
            REPLACEMENT_COST_M2.update(cached_base_costs)

        # Axis B: Vulnerability Parameter Shift
        for target_curves, curve_label in [(CURVES_STANDARD, 'curve_standard'), (CURVES_INFORMAL, 'curve_informal_adapted')]:
            loss_sensitivity_draws, _ = compute_scenario_loss_distribution(
                depth_samples_raw, current_types, current_areas, target_curves
            )
            sensitivity_records.append({
                'scenario': scenario, 'axis': 'curve', 'variant': curve_label,
                'mean_loss_usd': float(loss_sensitivity_draws.mean())
            })

    pd.DataFrame(sensitivity_records).to_csv(
        os.path.join(BASE_DIR, 'sensitivity_analysis_results.csv'), index=False
    )
    print("Saved: sensitivity_analysis_results.csv\n\nAll risk calculations successfully completed.")
