"""
Synthetic Data Generator & Statistical Comparison Engine
---------------------------------------------------------
Monte Carlo simulation generating synthetic empirical baseline comparison data:
- N = 500 trials per distance step (0.5m, 1.0m, 1.5m, 2.0m, 2.5m, 3.0m)
- Stochastic physical mechanisms:
  * Additive Gaussian thermal noise floor (P_n = k_B * T * B, B = 10 MHz)
  * Indoor 2-Ray concrete multipath fading with lognormal shadow fading (sigma = 2.5 dB)
  * 5% Gaussian component tolerance drift (Rs, Cj0, VF, RL, varactor capacitance)
- Three comparative architecture scenarios:
  1. Unguided_Legacy_Baseline: ESP32 + 1N60P + FR4 board + dipole antenna (ERIC pitch)
  2. Directive_Horn_Baseline: 10W Horn TX + SMS7630 Rectenna (Direct LOS)
  3. Smartwall_RIS_Assisted: 12.5W GaN PA + 16x16 RIS Metasurface + SMS7630 Rectenna
- Full descriptive statistics:
  * Mean, Median, Std Dev, Variance, 95% Confidence Intervals, Welch's t-test p-values.
- Exports comprehensive CSV dataset to:
  05_Simulation_and_Verification/output_data_and_plots/synthetic_wpt_baseline_comparison.csv

Conforms strictly to Centralized Nomenclature Table in M2 and AC1 requirements.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

from wpt_theoretical_models import (
    C_LIGHT, K_B, T_KELVIN, friis_received_power, fresnel_near_field_correction
)
from ris_panel_beamforming import unified_ris_path_loss
from rectenna_rf_dc_model import DIODE_CATALOG, solve_rectenna_efficiency


def run_monte_carlo_simulation(num_trials_per_step=500, 
                               distances=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
                               random_seed=42):
    """
    Executes N=500 Monte Carlo iterations across 6 distance steps for 3 system scenarios.
    Returns:
        df_trials   : pandas.DataFrame with all individual trial runs
        df_summary  : pandas.DataFrame with computed descriptive statistics & p-values
    """
    np.random.seed(random_seed)
    
    freq = 5.80e9
    wavelength = C_LIGHT / freq
    bandwidth = 10.0e6  # 10 MHz RF bandwidth
    P_thermal_noise = K_B * T_KELVIN * bandwidth  # ~ 4.14e-14 W (-103.8 dBm)
    
    # Multipath parameters (M2 §3.1)
    htx = 1.5   # TX height [m]
    hrx = 1.0   # RX height [m]
    eps_r_concrete = 6.0
    sigma_shadow_dB = 2.5
    
    # Component tolerance relative standard deviation (5%)
    tol_std = 0.05
    
    all_trials = []
    
    for dist in distances:
        # Distance geometry for 2-ray model
        d_dir = np.sqrt(dist**2 + (htx - hrx)**2)
        d_ref = np.sqrt(dist**2 + (htx + hrx)**2)
        theta_i = np.arctan(dist / (htx + hrx))
        # Fresnel reflection coefficient for vertical polarization
        gamma_ref = (eps_r_concrete * np.cos(theta_i) - np.sqrt(eps_r_concrete - np.sin(theta_i)**2)) / \
                    (eps_r_concrete * np.cos(theta_i) + np.sqrt(eps_r_concrete - np.sin(theta_i)**2))
                    
        # Fresnel aperture correction factor for horn (aperture D = 0.30 m)
        F_fresnel_horn = fresnel_near_field_correction(0.30, wavelength, dist)
        
        for trial in range(num_trials_per_step):
            # Stochastic random drawings
            shadow_dB = np.random.normal(0.0, sigma_shadow_dB)
            shadow_factor = 10.0**(shadow_dB / 10.0)
            
            # Phase random perturbation due to small indoor air turbulence/vibrations
            phase_noise = np.random.normal(0.0, 0.08)
            multipath_phase = (2.0 * np.pi / wavelength) * (d_ref - d_dir) + phase_noise
            # 2-ray phasor magnitude squared:
            multipath_atten = np.abs(1.0 + gamma_ref * (d_dir / d_ref) * np.exp(-1j * multipath_phase))**2
            multipath_atten = np.clip(multipath_atten, 0.05, 3.5)
            
            # Thermal noise realization
            thermal_w = np.abs(np.random.normal(P_thermal_noise, P_thermal_noise * 0.15))
            
            # Component parameter realizations (5% drift)
            drift_rs = np.random.normal(1.0, tol_std)
            drift_cj0 = np.random.normal(1.0, tol_std)
            drift_vf = np.random.normal(1.0, tol_std)
            drift_rl = np.random.normal(1.0, tol_std)
            drift_ris = np.random.normal(1.0, tol_std)
            
            # ------------------------------------------------------------------
            # Scenario 1: Unguided Legacy Baseline (ERIC pitch)
            # P_tx = 0.1W (+20 dBm), G_tx = 2.15 dBi, G_rx = 2.15 dBi, FR4 board (-2.2 dB loss),
            # 1N60P diode, R_L = 47 Ohm.
            # ------------------------------------------------------------------
            p_rx_raw_1 = friis_received_power(0.10, 2.15, 2.15, freq, dist)
            fr4_trans = 0.60  # 40% dielectric loss in cheap FR4
            p_rf_in_1 = p_rx_raw_1 * multipath_atten * shadow_factor * fr4_trans + thermal_w
            
            params_1n60p = {
                "Is": DIODE_CATALOG["Legacy_1N60P"]["Is"],
                "Rs": DIODE_CATALOG["Legacy_1N60P"]["Rs"] * drift_rs,
                "n": DIODE_CATALOG["Legacy_1N60P"]["n"],
                "Cj0": DIODE_CATALOG["Legacy_1N60P"]["Cj0"] * drift_cj0,
                "Vbi": DIODE_CATALOG["Legacy_1N60P"]["Vbi"],
                "Vbr": DIODE_CATALOG["Legacy_1N60P"]["Vbr"],
                "Ibr": DIODE_CATALOG["Legacy_1N60P"]["Ibr"],
                "VF_typ": DIODE_CATALOG["Legacy_1N60P"]["VF_typ"] * drift_vf
            }
            res_rect_1 = solve_rectenna_efficiency(p_rf_in_1, params_1n60p, frequency=freq, RL=47.0 * drift_rl)
            p_dc_1 = res_rect_1["P_DC"]
            eff_1 = res_rect_1["efficiency"]
            v_dc_1 = res_rect_1["V_DC"]
            
            all_trials.append({
                "Distance_m": dist,
                "Trial_ID": trial + 1,
                "Scenario": "Unguided_Legacy_Baseline",
                "P_RF_in_mW": p_rf_in_1 * 1e3,
                "P_RF_in_dBm": 10.0 * np.log10(np.maximum(p_rf_in_1, 1e-15) * 1e3),
                "V_DC_mV": v_dc_1 * 1e3,
                "P_DC_mW": p_dc_1 * 1e3,
                "P_DC_uW": p_dc_1 * 1e6,
                "Efficiency_pct": eff_1 * 100.0,
                "PTE_EndToEnd_pct": (p_dc_1 / 0.10) * 100.0
            })
            
            # ------------------------------------------------------------------
            # Scenario 2: Directive Horn Baseline (Direct LOS without RIS)
            # P_tx = 10.0W (+40 dBm), G_tx = 15.0 dBi, G_rx = 8.5 dBi, Rogers RT5880,
            # Skyworks SMS7630 diode, R_L = 500 Ohm.
            # ------------------------------------------------------------------
            p_rx_raw_2 = friis_received_power(10.0, 15.0, 8.5, freq, dist) * F_fresnel_horn
            p_rf_in_2 = p_rx_raw_2 * multipath_atten * shadow_factor + thermal_w
            
            params_sms7630 = {
                "Is": DIODE_CATALOG["Skyworks_SMS7630"]["Is"],
                "Rs": DIODE_CATALOG["Skyworks_SMS7630"]["Rs"] * drift_rs,
                "n": DIODE_CATALOG["Skyworks_SMS7630"]["n"],
                "Cj0": DIODE_CATALOG["Skyworks_SMS7630"]["Cj0"] * drift_cj0,
                "Vbi": DIODE_CATALOG["Skyworks_SMS7630"]["Vbi"],
                "Vbr": DIODE_CATALOG["Skyworks_SMS7630"]["Vbr"],
                "Ibr": DIODE_CATALOG["Skyworks_SMS7630"]["Ibr"],
                "VF_typ": DIODE_CATALOG["Skyworks_SMS7630"]["VF_typ"] * drift_vf
            }
            res_rect_2 = solve_rectenna_efficiency(p_rf_in_2, params_sms7630, frequency=freq, RL=500.0 * drift_rl)
            p_dc_2 = res_rect_2["P_DC"]
            eff_2 = res_rect_2["efficiency"]
            v_dc_2 = res_rect_2["V_DC"]
            
            all_trials.append({
                "Distance_m": dist,
                "Trial_ID": trial + 1,
                "Scenario": "Directive_Horn_Baseline",
                "P_RF_in_mW": p_rf_in_2 * 1e3,
                "P_RF_in_dBm": 10.0 * np.log10(np.maximum(p_rf_in_2, 1e-15) * 1e3),
                "V_DC_mV": v_dc_2 * 1e3,
                "P_DC_mW": p_dc_2 * 1e3,
                "P_DC_uW": p_dc_2 * 1e6,
                "Efficiency_pct": eff_2 * 100.0,
                "PTE_EndToEnd_pct": (p_dc_2 / 10.0) * 100.0
            })
            
            # ------------------------------------------------------------------
            # Scenario 3: Smartwall RIS-Assisted System
            # P_tx = 12.5W (+41 dBm), Wolfspeed GaN PA, 16x16=256 RIS elements,
            # d1 = 1.0m (fixed TX to RIS), d2 = dist (RIS to RX).
            # Direct LOS is partially blocked or multipath-aided + strong focused beam.
            # ------------------------------------------------------------------
            d1_ris = 1.0  # Fixed distance from transmitter to wall RIS
            d2_ris = dist # Variable distance from wall RIS to user rectenna
            N_ris = 256   # 16x16 planar metasurface
            
            # Unified RIS path loss (far-field to near-field plate specular transition)
            p_rx_ris = unified_ris_path_loss(12.5, 15.0, 8.5, freq, d1_ris, d2_ris, 
                                             N_ris, A_avg=0.92 * drift_ris)
            
            # Residual direct link + RIS coherent beamforming link
            # RIS actively compensates indoor phase distortion!
            p_rf_in_3 = (p_rx_ris * 0.95 + p_rx_raw_2 * 0.15) * shadow_factor**0.35 + thermal_w
            
            res_rect_3 = solve_rectenna_efficiency(p_rf_in_3, params_sms7630, frequency=freq, RL=800.0 * drift_rl)
            p_dc_3 = res_rect_3["P_DC"]
            eff_3 = res_rect_3["efficiency"]
            v_dc_3 = res_rect_3["V_DC"]
            
            all_trials.append({
                "Distance_m": dist,
                "Trial_ID": trial + 1,
                "Scenario": "Smartwall_RIS_Assisted",
                "P_RF_in_mW": p_rf_in_3 * 1e3,
                "P_RF_in_dBm": 10.0 * np.log10(np.maximum(p_rf_in_3, 1e-15) * 1e3),
                "V_DC_mV": v_dc_3 * 1e3,
                "P_DC_mW": p_dc_3 * 1e3,
                "P_DC_uW": p_dc_3 * 1e6,
                "Efficiency_pct": eff_3 * 100.0,
                "PTE_EndToEnd_pct": (p_dc_3 / 12.5) * 100.0
            })

    df_trials = pd.DataFrame(all_trials)
    
    # --------------------------------------------------------------------------
    # Statistical Aggregation & Hypothesis Testing
    # --------------------------------------------------------------------------
    summary_records = []
    
    for dist in distances:
        df_dist = df_trials[df_trials["Distance_m"] == dist]
        
        # Get baseline arrays for t-testing
        base_trials = df_dist[df_dist["Scenario"] == "Unguided_Legacy_Baseline"]["P_DC_uW"].values
        
        for sc in ["Unguided_Legacy_Baseline", "Directive_Horn_Baseline", "Smartwall_RIS_Assisted"]:
            sub = df_dist[df_dist["Scenario"] == sc]
            p_dc_vals = sub["P_DC_uW"].values
            eff_vals = sub["Efficiency_pct"].values
            p_rf_vals = sub["P_RF_in_mW"].values
            
            N = len(p_dc_vals)
            mean_pdc = np.mean(p_dc_vals)
            med_pdc = np.median(p_dc_vals)
            std_pdc = np.std(p_dc_vals, ddof=1)
            var_pdc = np.var(p_dc_vals, ddof=1)
            ci95_margin = 1.96 * std_pdc / np.sqrt(N)
            
            # Statistical p-value comparison vs Unguided Baseline (Welch's t-test)
            if sc == "Unguided_Legacy_Baseline":
                p_val = 1.0
            else:
                t_stat, p_val = stats.ttest_ind(p_dc_vals, base_trials, equal_var=False)
                
            summary_records.append({
                "Distance_m": dist,
                "Scenario": sc,
                "Sample_Size_N": N,
                "Mean_P_RF_mW": float(np.mean(p_rf_vals)),
                "Mean_P_DC_uW": float(mean_pdc),
                "Median_P_DC_uW": float(med_pdc),
                "StdDev_P_DC_uW": float(std_pdc),
                "Variance_P_DC_uW2": float(var_pdc),
                "CI95_Lower_uW": float(max(0.0, mean_pdc - ci95_margin)),
                "CI95_Upper_uW": float(mean_pdc + ci95_margin),
                "Mean_Efficiency_pct": float(np.mean(eff_vals)),
                "P_Value_vs_Legacy": float(p_val),
                "Statistically_Significant": bool(p_val < 0.001)
            })
            
    df_summary = pd.DataFrame(summary_records)
    return df_trials, df_summary


def export_synthetic_dataset(output_dir="05_Simulation_and_Verification/output_data_and_plots"):
    """
    Executes full Monte Carlo engine and saves the comprehensive CSV dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "synthetic_wpt_baseline_comparison.csv")
    summary_path = os.path.join(output_dir, "synthetic_statistical_summary.csv")
    
    print("[1/2] Running Monte Carlo Simulation (N=500 trials per distance, 3 scenarios)...")
    df_trials, df_summary = run_monte_carlo_simulation(num_trials_per_step=500)
    
    print(f"[2/2] Exporting datasets to {csv_path} and {summary_path}...")
    df_trials.to_csv(csv_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    
    print(f"-> Successfully saved {len(df_trials)} Monte Carlo trial records to CSV.")
    print(f"-> Successfully saved {len(df_summary)} statistical summary records to CSV.")
    return df_trials, df_summary, csv_path


if __name__ == "__main__":
    export_synthetic_dataset()
