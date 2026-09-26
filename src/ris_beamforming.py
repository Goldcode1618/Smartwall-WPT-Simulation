"""
RIS Metasurface Panel Beamforming Module
----------------------------------------
Scientific implementation of Reconfigurable Intelligent Surface (RIS) physics:
1. Unit cell complex reflection coefficient:
   - Gamma_n(theta_n) = A_n(theta_n) * exp(j * theta_n)
   - Amplitude-phase coupling loss model (Abeywickrama et al., 2020)
   - Discrete phase quantization (1-bit, 2-bit, 3-bit, and continuous)
2. Generalized Snell's law anomalous reflection:
   - Phase gradient dPhi/dx = k_0 * (sin(theta_r) - sin(theta_i))
   - Planar array factor AF(theta, psi)
3. Cascaded path loss and N^2 scaling:
   - Tang / Wu-Zhang far-field cascaded path loss model: P_rx proportional to N^2 / (d1 * d2)^2
   - Near-field aperture transition to specular plate reflection: P_rx proportional to 1 / (d1 + d2)^2
   - Unified physically consistent path loss bridging equation

Conforms strictly to Centralized Nomenclature Table in M2 (SW-WPT-M2-THEORY §4).
"""

import numpy as np

C_LIGHT = 2.99792458e8  # [m/s]


# ==============================================================================
# 1. UNIT CELL REFLECTION & COUPLING LOSS MODEL
# ==============================================================================

def unit_cell_amplitude_coupling(theta, A_min=0.841, phi_0=-np.pi/2.0, alpha=1.6):
    """
    Computes amplitude-dependent loss coupled with reflection phase shift.
    Formula (M2 §4.1 Eq. 543 / Abeywickrama et al. IEEE TWC 2020):
        A(theta) = (1 - A_min) * ((sin(theta - phi_0) + 1) / 2)^alpha + A_min
    """
    theta_arr = np.asarray(theta, dtype=float)
    sine_term = 0.5 * (np.sin(theta_arr - phi_0) + 1.0)
    A = (1.0 - A_min) * (sine_term**alpha) + A_min
    return A if theta_arr.ndim > 0 else float(A)


def quantize_phase(theta, bits=None):
    """
    Quantizes continuous phase shifts into discrete levels (M2 §4.1).
    """
    if bits is None or bits == 0 or bits == "inf":
        return theta
    
    num_states = 2**int(bits)
    step = 2.0 * np.pi / num_states
    
    theta_wrapped = np.mod(theta, 2.0 * np.pi)
    indices = np.round(theta_wrapped / step) % num_states
    quantized = indices * step
    
    return np.angle(np.exp(1j * quantized))


def unit_cell_reflection_coefficient(theta, A_min=0.841, bits=None):
    """
    Returns complex reflection coefficient Gamma_n = A_n * exp(j * theta_n).
    """
    theta_q = quantize_phase(theta, bits=bits)
    A = unit_cell_amplitude_coupling(theta_q, A_min=A_min)
    return A * np.exp(1j * theta_q)


# ==============================================================================
# 2. GENERALIZED SNELL'S LAW & BEAMFORMING ARRAY FACTOR
# ==============================================================================

def snell_phase_gradient(theta_i_rad, theta_r_rad, wavelength):
    """
    Computes required linear phase gradient along x-axis (M2 §4.2 Eq. 583):
        dPhi/dx = (2 * pi / lambda) * (sin(theta_r) - sin(theta_i))
    """
    k0 = 2.0 * np.pi / wavelength
    return k0 * (np.sin(theta_r_rad) - np.sin(theta_i_rad))


def generate_snell_phase_profile(Mx, My, dx, dy, wavelength, theta_i_rad, theta_r_rad):
    """
    Generates 2D phase distribution across planar RIS panel for anomalous reflection
    from incident elevation angle theta_i to reflected angle theta_r.
    """
    grad_x = snell_phase_gradient(theta_i_rad, theta_r_rad, wavelength)
    
    m_indices = np.arange(1, Mx + 1) - (Mx + 1.0) / 2.0
    x_coords = m_indices * dx
    
    phase_x = grad_x * x_coords
    phase_matrix = np.tile(phase_x[:, np.newaxis], (1, My))
    
    return np.angle(np.exp(1j * phase_matrix))


def compute_array_factor_2d(Mx, My, dx, dy, wavelength, phase_matrix, 
                            theta_grid_rad, psi_grid_rad, A_min=0.841):
    """
    Calculates spatial Array Factor AF(theta, psi) for the RIS panel (M2 §4.3 Eq. 595).
    """
    k0 = 2.0 * np.pi / wavelength
    m_indices = np.arange(1, Mx + 1) - (Mx + 1.0) / 2.0
    n_indices = np.arange(1, My + 1) - (My + 1.0) / 2.0
    
    x_m = m_indices * dx
    y_n = n_indices * dy
    
    A_matrix = unit_cell_amplitude_coupling(phase_matrix, A_min=A_min)
    gamma_matrix = A_matrix * np.exp(1j * phase_matrix)
    
    AF = np.zeros(theta_grid_rad.shape, dtype=complex)
    
    for m in range(Mx):
        for n in range(My):
            spatial_phase = k0 * np.sin(theta_grid_rad) * (
                x_m[m] * np.cos(psi_grid_rad) + y_n[n] * np.sin(psi_grid_rad)
            )
            AF += gamma_matrix[m, n] * np.exp(1j * spatial_phase)
            
    return AF


def compute_focusing_phase_profile(Mx, My, dx, dy, wavelength, r_tx, r_rx):
    """
    Computes coherent point-to-point 3D focusing phase profile (M2 §4.4 Eq. 614).
    """
    k0 = 2.0 * np.pi / wavelength
    m_indices = np.arange(1, Mx + 1) - (Mx + 1.0) / 2.0
    n_indices = np.arange(1, My + 1) - (My + 1.0) / 2.0
    
    x_grid, y_grid = np.meshgrid(m_indices * dx, n_indices * dy, indexing='ij')
    
    d1 = np.sqrt((x_grid - r_tx[0])**2 + (y_grid - r_tx[1])**2 + (0.0 - r_tx[2])**2)
    d2 = np.sqrt((r_rx[0] - x_grid)**2 + (r_rx[1] - y_grid)**2 + (r_rx[2] - 0.0)**2)
    
    phase_optimal = k0 * (d1 + d2)
    return np.angle(np.exp(1j * phase_optimal))


# ==============================================================================
# 3. CASCADED PATH LOSS & N^2 SCALING MODEL
# ==============================================================================

def far_field_cascaded_path_loss(P_tx, G_tx_dBi, G_rx_dBi, frequency, d1, d2, 
                                 N_elements, A_avg=0.92, G_e_dBi=4.93):
    """
    Tang / Wu-Zhang far-field cascaded path loss model (M2 §4.4 Eq. 619).
    """
    wavelength = C_LIGHT / frequency
    G_tx_lin = 10.0**(G_tx_dBi / 10.0)
    G_rx_lin = 10.0**(G_rx_dBi / 10.0)
    G_e_lin = 10.0**(G_e_dBi / 10.0)
    
    d1_arr = np.asarray(d1, dtype=float)
    d2_arr = np.asarray(d2, dtype=float)
    
    numerator = P_tx * G_tx_lin * G_rx_lin * (G_e_lin**2) * (wavelength**4) * ((N_elements * A_avg)**2)
    denominator = ((4.0 * np.pi)**4) * (d1_arr**2) * (d2_arr**2)
    
    P_rx = numerator / denominator
    return P_rx if (d1_arr.ndim > 0 or d2_arr.ndim > 0) else float(P_rx)


def near_field_specular_path_loss(P_tx, G_tx_dBi, G_rx_dBi, frequency, d1, d2, 
                                  eta_geom=0.85):
    """
    Near-field plate specular reflection asymptote (M2 §4.4 Eq. 631).
    """
    wavelength = C_LIGHT / frequency
    G_tx_lin = 10.0**(G_tx_dBi / 10.0)
    G_rx_lin = 10.0**(G_rx_dBi / 10.0)
    
    d1_arr = np.asarray(d1, dtype=float)
    d2_arr = np.asarray(d2, dtype=float)
    
    numerator = P_tx * G_tx_lin * G_rx_lin * (wavelength**2) * eta_geom
    denominator = ((4.0 * np.pi)**2) * ((d1_arr + d2_arr)**2)
    
    P_rx = numerator / denominator
    return P_rx if (d1_arr.ndim > 0 or d2_arr.ndim > 0) else float(P_rx)


def unified_ris_path_loss(P_tx, G_tx_dBi, G_rx_dBi, frequency, d1, d2, 
                          N_elements, A_avg=0.92, cell_area=None):
    """
    Unified path loss formulation bridging far-field cascaded (d1*d2)^(-2) 
    and near-field specular (d1 + d2)^(-2) transitions (Özdogan et al. / Tang et al.).
    """
    wavelength = C_LIGHT / frequency
    if cell_area is None:
        cell_area = (wavelength / 2.0)**2
        
    G_tx_lin = 10.0**(G_tx_dBi / 10.0)
    G_rx_lin = 10.0**(G_rx_dBi / 10.0)
    
    d1_arr = np.asarray(d1, dtype=float)
    d2_arr = np.asarray(d2, dtype=float)
    
    L_char = (N_elements * cell_area) / wavelength
    eff_distance = (d1_arr * d2_arr) / L_char + (d1_arr + d2_arr)
    
    P_rx = P_tx * G_tx_lin * G_rx_lin * ((wavelength / (4.0 * np.pi))**2) * (A_avg**2) / (eff_distance**2)
    return P_rx if (d1_arr.ndim > 0 or d2_arr.ndim > 0) else float(P_rx)
