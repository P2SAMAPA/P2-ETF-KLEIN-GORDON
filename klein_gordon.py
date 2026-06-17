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
    Using finite difference approximation on the return series.
    """
    if len(returns) < 5:
        return 0.0
    # Macro-dependent mass: m = mass_base + mass_range * macro_factor
    mass = mass_base + mass_range * macro_factor
    # Discretise space: use indices as x
    n = len(returns)
    dx = 1.0
    dt = 1.0
    # Initial conditions: φ(x,0) = returns[x], φ_t(x,0) = 0 (momentum zero)
    phi = np.zeros((n, n))  # time x space
    phi[0, :] = returns
    # First time step
    phi[1, 1:n-1] = phi[0, 1:n-1] + 0.5 * dt**2 * (
        (phi[0, 2:] - 2*phi[0, 1:n-1] + phi[0, :-2]) / dx**2 - mass**2 * phi[0, 1:n-1]
    )
    # Boundary conditions: zero at edges (Dirichlet)
    phi[1, 0] = 0
    phi[1, n-1] = 0
    # Time evolution (leapfrog method)
    for t in range(1, n-1):
        # Interior points
        phi[t+1, 1:n-1] = 2*phi[t, 1:n-1] - phi[t-1, 1:n-1] + dt**2 * (
            (phi[t, 2:] - 2*phi[t, 1:n-1] + phi[t, :-2]) / dx**2 - mass**2 * phi[t, 1:n-1]
        )
        # Boundaries
        phi[t+1, 0] = 0
        phi[t+1, n-1] = 0
    # Score = field amplitude at the last time step (absolute value)
    field_amplitude = np.abs(phi[-1])
    # Return the amplitude at the last spatial point (or average)
    # We'll use the amplitude at the last point as the score for the ETF
    score = field_amplitude[-1] if len(field_amplitude) > 0 else 0.0
    return float(score)
