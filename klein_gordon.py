import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def compute_macro_factor(macro_df):
    """Compute composite macro factor from all macro variables."""
    if len(macro_df) < 2:
        return np.ones(len(macro_df)) * 0.5
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    pca = PCA(n_components=1)
    factor = pca.fit_transform(macro_scaled).flatten()
    factor = (factor - factor.min()) / (factor.max() - factor.min() + 1e-8)
    return factor

def klein_gordon_field(returns, macro_factor, mass_base=1.0, mass_range=0.5):
    """
    Solve Klein-Gordon equation: (∂²/∂t² - ∂²/∂x² + m²) φ = 0
    Returns the field amplitude (energy density) at the last time step.
    """
    if len(returns) < 5:
        return 0.0
    # Macro-dependent mass
    mass = mass_base + mass_range * macro_factor
    n = len(returns)
    dx = 1.0
    dt = 0.5  # reduced time step for stability
    # Initial field and momentum
    phi = returns
    phi_prev = returns * 0.0  # initial velocity zero
    # Store fields for energy calculation
    phi_t = phi.copy()
    phi_t_prev = phi_prev.copy()
    # Time evolution
    for t in range(1, n):
        # Compute spatial second derivative (laplacian) using finite differences
        laplacian = np.zeros(n)
        laplacian[1:-1] = (phi_t[2:] - 2*phi_t[1:-1] + phi_t[:-2]) / dx**2
        # Boundaries: zero derivative (Neumann)
        laplacian[0] = (phi_t[1] - phi_t[0]) / dx**2
        laplacian[-1] = (phi_t[-2] - phi_t[-1]) / dx**2
        # Klein-Gordon time evolution (leapfrog)
        phi_next = 2*phi_t - phi_t_prev + dt**2 * (laplacian - mass**2 * phi_t)
        phi_t_prev = phi_t
        phi_t = phi_next
    # Compute energy density at the last step: E = 0.5*(∂φ/∂t)^2 + 0.5*(∇φ)^2 + 0.5*m^2*φ^2
    # We'll use the amplitude (absolute value) of the field at the last point
    # as a simple proxy for energy
    amplitude = np.abs(phi_t[-1]) if len(phi_t) > 0 else 0.0
    return float(amplitude)
