"""
Nonlinear Schottky Diode Rectenna Module
----------------------------------------
Scientific implementation of rectenna RF-to-DC conversion physics:
1. SPICE Diode Nonlinear Model:
   - Shockley equation with series resistance: I_D(V_j) = I_s * [exp(q*V_j / (n*k_B*T)) - 1]
   - Voltage-dependent depletion junction capacitance: C_j(V_j)
   - Ohmic series resistance loss: P_loss_Rs = I_rms^2 * R_s
   - Reverse breakdown avalanche: V_rev > V_br
2. Harmonic Balance & Yoo-Chang / McSpadden Analytical Solver:
   - Dynamic half-cycle conduction angle: cos(theta_on) = (V_F + V_DC) / V_0
   - Full 3-regime bell-shaped efficiency curve:
     * Regime 1: Sub-threshold square-law (P_in < -20 dBm, eta ~ 1% - 15%)
     * Regime 2: Large-signal linear peak (-10 dBm to +20 dBm, eta ~ 60% - 85%)
     * Regime 3: Reverse breakdown and thermal saturation (P_in > +20 dBm, collapse)
3. Load Resistance Optimization (R_L sweep from 10 Ohm to 100 kOhm).

Conforms strictly to Centralized Nomenclature Table in M2 (SW-WPT-M2-THEORY §5).
"""

import numpy as np

# Physical constants
K_B = 1.380649e-23
Q_E = 1.602176634e-19
T_KELVIN = 300.0
V_T = (K_B * T_KELVIN) / Q_E  # ~ 25.86 mV


# ==============================================================================
# 1. MANUFACTURER DIODE PARAMETER DATABASE
# ==============================================================================

DIODE_CATALOG = {
    "Skyworks_SMS7630": {
        "name": "Skyworks SMS7630-079LF (Low-Barrier Zero-Bias)",
        "Is": 5.0e-6,      # [A]
        "Rs": 20.0,        # [Ohm]
        "n": 1.05,         # Ideality factor
        "Cj0": 0.14e-12,   # [F] (0.14 pF)
        "Vbi": 0.34,       # [V] Built-in potential
        "gamma": 0.40,     # Grading coefficient
        "Fc": 0.50,        # Forward capacitance coefficient
        "Vbr": 2.0,        # [V] Reverse breakdown voltage
        "Ibr": 1.0e-4,     # [A] Breakdown current
        "VF_typ": 0.120,   # [V] Forward voltage @ 0.1 mA
        "fc": 56.84e9      # [Hz] 56.84 GHz theoretical cutoff
    },
    "Broadcom_HSMS2850": {
        "name": "Broadcom HSMS-2850 (Zero-Bias Detector)",
        "Is": 3.0e-6,
        "Rs": 25.0,
        "n": 1.06,
        "Cj0": 0.18e-12,
        "Vbi": 0.35,
        "gamma": 0.50,
        "Fc": 0.50,
        "Vbr": 3.8,
        "Ibr": 3.0e-4,
        "VF_typ": 0.200,
        "fc": 35.37e9
    },
    "Broadcom_HSMS2860": {
        "name": "Broadcom HSMS-2860 (High-Power High-Barrier)",
        "Is": 5.0e-8,
        "Rs": 5.5,
        "n": 1.08,
        "Cj0": 0.18e-12,
        "Vbi": 0.65,
        "gamma": 0.50,
        "Fc": 0.50,
        "Vbr": 7.0,
        "Ibr": 1.0e-5,
        "VF_typ": 0.300,
        "fc": 160.76e9
    },
    "Legacy_1N60P": {
        "name": "Legacy 1N60P (Flawed Point-Contact Diode)",
        "Is": 1.0e-7,
        "Rs": 30.0,
        "n": 1.35,
        "Cj0": 1.50e-12,   # 1.50 pF (Massive capacitive shunting)
        "Vbi": 0.40,
        "gamma": 0.50,
        "Fc": 0.50,
        "Vbr": 4.0,
        "Ibr": 1.0e-5,
        "VF_typ": 0.350,   # High forward threshold
        "fc": 3.54e9       # Collapses at microwave frequencies
    }
}


# ==============================================================================
# 2. SPICE DIODE INTERNAL EQUATIONS
# ==============================================================================

def diode_cutoff_frequency(Rs, Cj0):
    """
    Theoretical cut-off frequency of Schottky diode (M2 §5.3 Eq. 698):
        f_c = 1 / (2 * pi * Rs * Cj0)
    """
    return 1.0 / (2.0 * np.pi * Rs * Cj0)


def dynamic_junction_capacitance(Vj, Cj0, Vbi, gamma=0.5, Fc=0.5):
    """
    SPICE voltage-dependent junction depletion capacitance C_j(V_j) (M2 §5.3 Eq. 695).
    """
    Vj_arr = np.asarray(Vj, dtype=float)
    V_thresh = Fc * Vbi
    
    C_sub = Cj0 / ((1.0 - Vj_arr / Vbi)**gamma)
    
    coeff = Cj0 / ((1.0 - Fc)**(1.0 + gamma))
    C_fwd = coeff * (1.0 - Fc * (1.0 + gamma) + gamma * (Vj_arr / Vbi))
    
    C = np.where(Vj_arr < V_thresh, C_sub, C_fwd)
    return C if Vj_arr.ndim > 0 else float(C)


def safe_breakdown_power_threshold(Vbr, RL):
    """
    Maximum safe incident RF power before reverse avalanche breakdown (M2 §5.3 Eq. 710):
        P_in,max = V_br^2 / (8 * RL)
    """
    return (Vbr**2) / (8.0 * RL)


# ==============================================================================
# 3. HARMONIC BALANCE & 3-REGIME EFFICIENCY SOLVER
# ==============================================================================

def solve_rectenna_efficiency(P_in_W, diode_params, frequency=5.8e9, 
                              RL=500.0, Rin=50.0, eta_match=0.95):
    """
    Solves RF-to-DC conversion efficiency across all 3 regimes:
    Regime 1: Sub-threshold (Shockley square-law expansion)
    Regime 2: Large-signal linear peak (Yoo-Chang / McSpadden model)
    Regime 3: Reverse breakdown avalanche & thermal collapse
    """
    P_in = np.asarray(P_in_W, dtype=float)
    scalar_input = (P_in.ndim == 0)
    P_in = np.atleast_1d(P_in)
    
    Is = diode_params["Is"]
    Rs = diode_params["Rs"]
    n_dio = diode_params["n"]
    Cj0 = diode_params["Cj0"]
    Vbi = diode_params["Vbi"]
    Vbr = diode_params["Vbr"]
    Ibr = diode_params["Ibr"]
    VF = diode_params["VF_typ"]
    
    omega_0 = 2.0 * np.pi * frequency
    fc = diode_cutoff_frequency(Rs, Cj0)
    
    V_0 = np.sqrt(2.0 * np.maximum(P_in, 1e-18) * Rin)
    
    V_DC = np.zeros_like(P_in)
    I_DC = np.zeros_like(P_in)
    P_DC = np.zeros_like(P_in)
    P_loss_Rs = np.zeros_like(P_in)
    eff = np.zeros_like(P_in)
    regimes = np.zeros_like(P_in, dtype=int)
    
    V_turn_on = VF * 0.60  # Low-barrier Schottky conduction onset (M3 Table 3.1: 60-120 mV)
    
    for idx, p_w in enumerate(P_in):
        v0 = V_0[idx]
        
        if v0 < V_turn_on:
            # ==============================================================
            # REGIME 1: SUB-THRESHOLD / SQUARE-LAW REGIME (M2 §5.5 Eq. 779-781)
            # v0 << VF, Shockley curvature governs detection: I_DC proportional to P_in
            # ==============================================================
            regimes[idx] = 1
            # Current responsivity: beta_0 = 1 / (2 * n * V_T * (1 + (omega_0 * Rs * Cj0)^2))
            beta_0 = 1.0 / (2.0 * n_dio * V_T * (1.0 + (omega_0 * Rs * Cj0)**2))
            # Delivered power to diode junction
            i_dc = beta_0 * p_w * eta_match * (Rin / (Rs + RL))
            v_dc = i_dc * RL
            p_dc = (v_dc**2) / RL
            
            # Efficiency scales linearly with input power
            eta = min(p_dc / (p_w + 1e-18), 0.35) * eta_match
            p_dc = p_w * eta
            v_dc = np.sqrt(p_dc * RL)
            i_dc = v_dc / RL if RL > 0 else 0.0
            p_loss = (i_dc**2) * Rs
            
        elif v0 <= Vbr * 0.85:
            # ==============================================================
            # REGIME 2: LARGE-SIGNAL LINEAR PEAK RECTIFICATION
            # Diode conduction pulse into reservoir capacitor C_L (M2 §5.4-5.5)
            # ==============================================================
            regimes[idx] = 2
            
            # Root-finding for x = cos(theta_on) in [V_turn_on / v0, 0.99]
            # Equating average diode pulse current to load current V_DC / RL:
            # x = V_turn_on / v0 + (RL / (pi * Rs)) * [sqrt(1 - x^2) - x * arccos(x)]
            x_min = max(0.05, V_turn_on / (v0 + 1e-12))
            x_lo = x_min
            x_hi = 0.99
            for _ in range(16):
                x_mid = 0.5 * (x_lo + x_hi)
                th = np.arccos(np.clip(x_mid, 0.0, 1.0))
                g_val = (RL / (np.pi * Rs)) * (np.sin(th) - th * x_mid) + (V_turn_on / v0)
                if g_val > x_mid:
                    x_lo = x_mid
                else:
                    x_hi = x_mid
            x_opt = 0.5 * (x_lo + x_hi)
            theta_on = np.arccos(np.clip(x_opt, 0.05, 0.98))
            
            v_dc_raw = max(1e-6, v0 * x_opt - V_turn_on)
            
            # Yoo-Chang / McSpadden analytical efficiency formula (M2 §5.5 Eq. 756):
            Cj_dyn = dynamic_junction_capacitance(VF * 0.5, Cj0, Vbi)
            theta_ratio = theta_on / np.pi
            
            denom = (1.0 + theta_ratio * (Rs / RL) 
                     + (VF / (v_dc_raw + 1e-12)) * (1.0 + theta_ratio * (Rs / RL))
                     + ((Cj_dyn**2) * (omega_0**2) * Rs * RL) / (theta_ratio + 1e-12)
                     + (frequency / fc)**2)
                     
            eta = np.clip(eta_match / denom, 0.0, 0.88)
            p_dc = p_w * eta
            v_dc = np.sqrt(p_dc * RL)
            i_dc = v_dc / RL if RL > 0 else 0.0
            
            i_rms = i_dc * np.sqrt(np.pi / (theta_on + 1e-12))
            p_loss = (i_rms**2) * Rs
            
        else:
            regimes[idx] = 3
            v_dc_nom = v0 * 0.7
            v_rev = v0 + v_dc_nom
            
            delta_v = np.clip(v_rev - Vbr, 0.0, 5.0)
            i_rev = Ibr * np.expm1(delta_v / (n_dio * V_T * 4.0))
            
            i_dc_fwd = (v0 - VF) / (Rs + RL)
            i_dc = np.maximum(0.0, i_dc_fwd - i_rev)
            v_dc = i_dc * RL
            p_dc = (v_dc**2) / RL
            
            i_rms = np.sqrt(i_dc_fwd**2 + i_rev**2)
            p_loss = (i_rms**2) * Rs
            
            breakdown_decay = np.exp(-1.5 * delta_v)
            eta = (p_dc / (p_w + 1e-12)) * breakdown_decay * eta_match
            
        eff[idx] = min(max(float(eta), 0.0), 1.0)
        P_DC[idx] = p_w * eff[idx]
        v_dc = np.sqrt(P_DC[idx] * RL)
        i_dc = v_dc / RL if RL > 0 else 0.0
        V_DC[idx] = v_dc
        I_DC[idx] = i_dc
        P_loss_Rs[idx] = p_loss
        
    res = {
        "V_0": V_0 if not scalar_input else float(V_0[0]),
        "V_DC": V_DC if not scalar_input else float(V_DC[0]),
        "I_DC": I_DC if not scalar_input else float(I_DC[0]),
        "P_DC": P_DC if not scalar_input else float(P_DC[0]),
        "P_loss_Rs": P_loss_Rs if not scalar_input else float(P_loss_Rs[0]),
        "efficiency": eff if not scalar_input else float(eff[0]),
        "regime": regimes if not scalar_input else int(regimes[0])
    }
    return res


# ==============================================================================
# 4. LOAD RESISTANCE OPTIMIZATION
# ==============================================================================

def optimize_load_resistance(P_in_W, diode_params, frequency=5.8e9, 
                             RL_min=20.0, RL_max=50000.0, num_points=100):
    """
    Sweeps load resistance RL to find optimal resistance RL_opt that maximizes
    RF-to-DC conversion efficiency at given input power level (M2 §5.6 Eq. 793).
    """
    RL_sweep = np.logspace(np.log10(RL_min), np.log10(RL_max), num_points)
    eff_list = []
    
    for rl in RL_sweep:
        res = solve_rectenna_efficiency(P_in_W, diode_params, frequency=frequency, RL=rl)
        eff_list.append(res["efficiency"])
        
    eff_arr = np.array(eff_list)
    best_idx = np.argmax(eff_arr)
    
    return {
        "RL_sweep": RL_sweep,
        "efficiencies": eff_arr,
        "RL_opt": float(RL_sweep[best_idx]),
        "max_efficiency": float(eff_arr[best_idx])
    }
