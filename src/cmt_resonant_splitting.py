"""
WPT Theoretical Models Module
-----------------------------
Scientific implementation of multi-frequency Wireless Power Transfer physics:
1. Near-field magnetic resonance (85 kHz, 6.78 MHz, 13.56 MHz):
   - Neumann mutual inductance with complete elliptic integrals (ellipk, ellipe)
   - Asymptotic dipole validation (1/d^3)
   - Coupled-Mode Theory (CMT) differential modes and Figure of Merit (U)
   - Frequency splitting bifurcation (omega_even, omega_odd) in over-coupled regime
   - Series-Series (SS) topology impedance and efficiency vs distance
2. Far-field radiative microwave (2.45 GHz, 5.8 GHz):
   - Maxwell-Poynting power density vector
   - Friis transmission equation
   - Rayleigh and Reactive boundary calculations
   - Radiating near-field (Fresnel) correction factor for laboratory testbenches (0.5m - 3.0m)
   - ICNIRP 2020 / IEEE C95.1-2019 human safety threshold verification

Conforms strictly to Centralized Nomenclature Table in M2 (SW-WPT-M2-THEORY).
"""

import numpy as np
from scipy.special import ellipk, ellipe

# ==============================================================================
# 1. UNIVERSAL PHYSICAL CONSTANTS (M2 Nomenclature Table)
# ==============================================================================
C_LIGHT = 2.99792458e8         # Speed of light in vacuum [m/s]
MU_0 = 4.0 * np.pi * 1e-7      # Magnetic permeability of vacuum [H/m]
EPS_0 = 8.854187817e-12        # Electric permittivity of vacuum [F/m]
ETA_0 = np.sqrt(MU_0 / EPS_0)  # Free-space wave impedance ~ 376.73 [Ohm]
K_B = 1.380649e-23             # Boltzmann constant [J/K]
Q_E = 1.602176634e-19          # Elementary electron charge [C]
T_KELVIN = 300.0               # Standard ambient room temperature [K]
V_THERMAL = (K_B * T_KELVIN) / Q_E  # Thermal equivalent voltage ~ 25.86 [mV]


# ==============================================================================
# 2. NEAR-FIELD MAGNETIC RESONANCE (85 kHz, 6.78 MHz, 13.56 MHz)
# ==============================================================================

def neumann_mutual_inductance(r1, r2, d, N1=1, N2=1, rw=1.0e-3):
    """
    Computes exact mutual inductance between two coaxial circular loops
    using Neumann's formula and complete elliptic integrals of 1st and 2nd kind.
    
    Formula (M2 §2.1 Eq. 181):
        k_m = sqrt( 4 * r1 * r2 / ((r1 + r2)^2 + d^2 + rw^2) )
        m = k_m^2
        M = mu_0 * sqrt(r1 * r2) * [ (2/k_m - k_m)*K(m) - (2/k_m)*E(m) ]
        M_total = N1 * N2 * M
    """
    d_arr = np.asarray(d, dtype=float)
    denom = (r1 + r2)**2 + d_arr**2 + rw**2
    k_m = np.sqrt(4.0 * r1 * r2 / denom)
    
    k_m = np.clip(k_m, 1e-12, 0.999999999999)
    m = k_m**2  # SciPy ellipk/ellipe take parameter m = k^2
    
    K_val = ellipk(m)
    E_val = ellipe(m)
    
    term = (2.0 / k_m - k_m) * K_val - (2.0 / k_m) * E_val
    M_single = MU_0 * np.sqrt(r1 * r2) * term
    
    M_total = N1 * N2 * M_single
    return M_total if d_arr.ndim > 0 else float(M_total)


def neumann_dipole_asymptote(r1, r2, d, N1=1, N2=1):
    """
    Asymptotic magnetic dipole approximation for mutual inductance (d >> r1, r2).
    Formula (M2 §2.1 Eq. 195):
        M_dipole = (mu_0 * pi * N1 * N2 * r1^2 * r2^2) / (2 * d^3)
    """
    d_arr = np.asarray(d, dtype=float)
    M_dipole = (MU_0 * np.pi * N1 * N2 * (r1**2) * (r2**2)) / (2.0 * (d_arr**3))
    return M_dipole if d_arr.ndim > 0 else float(M_dipole)


def self_inductance_circular_coil(r, N=1, rw=1.0e-3):
    """
    Computes self-inductance of a circular loop/coil with finite wire radius rw.
    L = mu_0 * N^2 * r * [ ln(8 * r / rw) - 2.0 ]
    """
    return float(MU_0 * (N**2) * r * (np.log(8.0 * r / rw) - 2.0))


def cmt_parameters(omega_0, M, L1, L2, R1, R2, RL):
    """
    Coupled-Mode Theory (CMT) differential rate equations parameters:
        k_mag = M / sqrt(L1 * L2)
        kappa = (omega_0 * M) / (2 * sqrt(L1 * L2)) = 0.5 * k_mag * omega_0
        Gamma_1 = R1 / (2 * L1) = omega_0 / (2 * Q1)
        Gamma_2 = R2 / (2 * L2) = omega_0 / (2 * Q2)
        Gamma_L = RL / (2 * L2)
        Q1 = omega_0 * L1 / R1
        Q2 = omega_0 * L2 / R2
        U = kappa / sqrt(Gamma_1 * Gamma_2) = k_mag * sqrt(Q1 * Q2)
    """
    k_mag = M / np.sqrt(L1 * L2)
    kappa = 0.5 * k_mag * omega_0
    Gamma_1 = R1 / (2.0 * L1)
    Gamma_2 = R2 / (2.0 * L2)
    Gamma_L = RL / (2.0 * L2)
    Q1 = omega_0 * L1 / R1
    Q2 = omega_0 * L2 / R2
    U = k_mag * np.sqrt(Q1 * Q2)
    
    return {
        "k_mag": k_mag,
        "kappa": kappa,
        "Gamma_1": Gamma_1,
        "Gamma_2": Gamma_2,
        "Gamma_L": Gamma_L,
        "Q1": Q1,
        "Q2": Q2,
        "U": U
    }


def cmt_theoretical_max_efficiency(U):
    """
    Upper theoretical bound on magnetic resonance PTE from CMT (M2 §2.1 Eq. 245):
        eta_max = U^2 / (1 + sqrt(1 + U^2))^2
    """
    U_arr = np.asarray(U, dtype=float)
    eta = (U_arr**2) / ((1.0 + np.sqrt(1.0 + U_arr**2))**2)
    return eta if U_arr.ndim > 0 else float(eta)


def frequency_splitting_bifurcation(omega_0, k_mag, Q1, Q2):
    """
    Computes eigenmode bifurcation for over-coupled regime (M2 §2.1 Eq. 257-263):
        k_c = 1 / sqrt(Q1 * Q2)
        If k_mag > k_c:
            omega_even = omega_0 / sqrt(1 + k_mag)
            omega_odd  = omega_0 / sqrt(1 - k_mag)
    """
    k_c = 1.0 / np.sqrt(Q1 * Q2)
    k_arr = np.asarray(k_mag, dtype=float)
    
    omega_even = np.where(k_arr > k_c, omega_0 / np.sqrt(1.0 + k_arr), omega_0)
    safe_k = np.clip(k_arr, 0.0, 0.99)
    omega_odd = np.where(k_arr > k_c, omega_0 / np.sqrt(1.0 - safe_k), omega_0)
    
    return {
        "k_critical": k_c,
        "omega_even": omega_even if k_arr.ndim > 0 else float(omega_even),
        "omega_odd": omega_odd if k_arr.ndim > 0 else float(omega_odd)
    }


def series_series_efficiency(omega, M, L1, C1, R1, L2, C2, R2, RL):
    """
    Computes exact circuit impedance and power transfer efficiency for
    Series-Series (SS) resonant topology (M2 §2.1 Eq. 284-306).
    """
    omega_arr = np.asarray(omega, dtype=float)
    M_arr = np.asarray(M, dtype=float)
    
    X1 = omega_arr * L1 - 1.0 / (omega_arr * C1)
    X2 = omega_arr * L2 - 1.0 / (omega_arr * C2)
    Z1 = R1 + 1j * X1
    Z2 = R2 + 1j * X2
    
    Z_ref = ((omega_arr * M_arr)**2) / (Z2 + RL)
    Z_in = Z1 + Z_ref
    
    num = (omega_arr**2) * (M_arr**2) * RL
    denom = np.abs(Z2 + RL)**2 * np.real(Z_in)
    eta = num / denom
    
    return {
        "Z_ref": Z_ref,
        "Z_in": Z_in,
        "efficiency": np.clip(np.real(eta), 0.0, 1.0)
    }


# ==============================================================================
# 3. FAR-FIELD RADIATIVE MICROWAVE (2.45 GHz, 5.8 GHz)
# ==============================================================================

def rayleigh_distance(D, wavelength):
    """
    Computes Fraunhofer / Rayleigh far-field boundary distance:
        R_Rayleigh = 2 * D^2 / lambda
    """
    return 2.0 * (D**2) / wavelength


def reactive_distance(D, wavelength):
    """
    Computes outer boundary of reactive near-field:
        R_reactive = 0.62 * sqrt(D^3 / lambda)
    """
    return 0.62 * np.sqrt((D**3) / wavelength)


def poynting_power_density(P_tx, G_tx_dBi, distance):
    """
    Calculates time-averaged Poynting vector flux power density (M2 §2.2 Eq. 360):
        S_inc = (P_tx * G_tx) / (4 * pi * d^2) = EIRP / (4 * pi * d^2)
    """
    d_arr = np.asarray(distance, dtype=float)
    G_tx_lin = 10.0**(G_tx_dBi / 10.0)
    EIRP = P_tx * G_tx_lin
    S_inc = EIRP / (4.0 * np.pi * (d_arr**2))
    return S_inc if d_arr.ndim > 0 else float(S_inc)


def friis_received_power(P_tx, G_tx_dBi, G_rx_dBi, frequency, distance):
    """
    Classical Friis transmission equation (M2 §2.2 Eq. 370):
        P_rx = P_tx * G_tx * G_rx * (lambda / (4 * pi * d))^2
    """
    d_arr = np.asarray(distance, dtype=float)
    wavelength = C_LIGHT / frequency
    G_tx_lin = 10.0**(G_tx_dBi / 10.0)
    G_rx_lin = 10.0**(G_rx_dBi / 10.0)
    
    path_loss_factor = (wavelength / (4.0 * np.pi * d_arr))**2
    P_rx = P_tx * G_tx_lin * G_rx_lin * path_loss_factor
    return P_rx if d_arr.ndim > 0 else float(P_rx)


def fresnel_near_field_correction(D, wavelength, distance):
    """
    Aperture phase defocusing correction factor in the radiating near-field
    (Fresnel zone: R_reactive <= d < R_Rayleigh).
        F_corr(d) = 1 / sqrt(1 + (R_Rayleigh / (4 * d))^2)
    """
    d_arr = np.asarray(distance, dtype=float)
    R_R = rayleigh_distance(D, wavelength)
    F_corr = 1.0 / np.sqrt(1.0 + (R_R / (4.0 * d_arr))**2)
    return F_corr if d_arr.ndim > 0 else float(F_corr)


def fresnel_received_power(P_tx, G_tx_dBi, G_rx_dBi, frequency, distance, D_tx):
    """
    Fresnel-corrected received power on laboratory testbenches (M2 §2.3):
        P_rx,Fresnel = P_rx,Friis * F_corr(d)
    """
    wavelength = C_LIGHT / frequency
    P_rx_friis = friis_received_power(P_tx, G_tx_dBi, G_rx_dBi, frequency, distance)
    F_corr = fresnel_near_field_correction(D_tx, wavelength, distance)
    return P_rx_friis * F_corr


def icnirp_safety_audit(P_tx, G_tx_dBi, distance, S_lim=10.0):
    """
    Verifies human exposure compliance against ICNIRP 2020 / IEEE C95.1-2019 (M2 §3.2):
        S_inc <= S_lim (10.0 W/m^2 public reference limit at 2.45 - 5.8 GHz).
        r_safe = sqrt( EIRP / (4 * pi * S_lim) )
    """
    d_arr = np.asarray(distance, dtype=float)
    G_tx_lin = 10.0**(G_tx_dBi / 10.0)
    EIRP = P_tx * G_tx_lin
    S_inc = poynting_power_density(P_tx, G_tx_dBi, d_arr)
    r_safe = np.sqrt(EIRP / (4.0 * np.pi * S_lim))
    is_safe = S_inc <= S_lim
    
    return {
        "EIRP_W": EIRP,
        "EIRP_dBm": 10.0 * np.log10(EIRP * 1e3),
        "S_inc": S_inc,
        "S_lim": S_lim,
        "r_safe_meters": float(r_safe),
        "is_safe": is_safe
    }


# ==============================================================================
# 4. MULTI-FREQUENCY STANDARD TESTBENCH CONFIGURATIONS
# ==============================================================================

NEAR_FIELD_CONFIGS = {
    "85kHz": {
        "freq": 85.0e3,
        "r1": 0.15,
        "r2": 0.15,
        "N1": 15,
        "N2": 15,
        "rw": 1.5e-3,
        "R1": 0.25,
        "R2": 0.25,
        "RL": 10.0
    },
    "6.78MHz": {
        "freq": 6.78e6,
        "r1": 0.10,
        "r2": 0.10,
        "N1": 4,
        "N2": 4,
        "rw": 1.0e-3,
        "R1": 0.35,
        "R2": 0.35,
        "RL": 50.0
    },
    "13.56MHz": {
        "freq": 13.56e6,
        "r1": 0.08,
        "r2": 0.08,
        "N1": 3,
        "N2": 3,
        "rw": 1.0e-3,
        "R1": 0.45,
        "R2": 0.45,
        "RL": 50.0
    }
}

FAR_FIELD_CONFIGS = {
    "2.45GHz": {
        "freq": 2.45e9,
        "P_tx": 10.0,
        "G_tx_dBi": 12.0,
        "G_rx_dBi": 7.0,
        "D_tx": 0.35
    },
    "5.8GHz": {
        "freq": 5.80e9,
        "P_tx": 12.5,
        "G_tx_dBi": 15.0,
        "G_rx_dBi": 8.5,
        "D_tx": 0.30
    }
}
