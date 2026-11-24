import numpy as np

from rmc_discord.pairs import PairTerms
from rmc_discord.pairs.realspace import delta_pdf_slab


def test_slab_average_of_anisotropic_gaussians(tetragonal):
    """Compare with brute-force 3D sampling of the Gaussian densities."""

    U = np.array(
        [[0.02, 0.006, 0.004], [0.006, 0.03, -0.005], [0.004, -0.005, 0.025]]
    )
    d = np.array([[0.1, -0.05, 0.02], [0.05, -0.1, 0.0], [0.15, 0.0, 0.03]])
    w = np.array([2.0, -1.0, 0.5])
    Us = np.stack([U, 0.5 * U, U.T @ U * 20])
    terms = PairTerms(d, w, Us)
    x = np.linspace(0.0, 0.2, 41)
    y = np.linspace(-0.15, 0.05, 41)
    z_range = (-0.01, 0.03)
    pdf = delta_pdf_slab(terms, tetragonal, x, y, z_range)

    A = tetragonal.A
    z = np.linspace(*z_range, 401)
    X, Y, Z = np.meshgrid(x, y, z, indexing="xy")
    rho = 0
    for dp, wp, Up in zip(d, w, Us):
        Uinv = np.linalg.inv(Up)
        norm = wp / np.sqrt((2 * np.pi) ** 3 * np.linalg.det(Up))
        r = (np.stack([X, Y, Z], -1) - dp) @ A.T
        quad = np.einsum("...a,ab,...b->...", r, Uinv, r)
        rho = rho + norm * np.exp(-0.5 * quad)
    expected = np.trapezoid(rho, z, axis=-1) / (z_range[1] - z_range[0])
    assert np.allclose(
        pdf, expected, rtol=1e-4, atol=1e-6 * abs(expected).max()
    )
