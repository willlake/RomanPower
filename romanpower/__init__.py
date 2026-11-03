def _set_xla_p2p_flag():
    """
    Several GPUs in one process (``create_sharding_mesh()`` over all the local GPUs, without ``jax.distributed``): XLA must run the
    single-process all-to-alls and collective-permutes (jaxdecomp's FFT transposes, the halo exchanges of the painting) as
    peer-to-peer copies behind a host rendezvous, not as NCCL kernels. With NCCL kernels the painting deadlocks intermittently, two
    GPUs at 100 % and the others idle (upstream jax-power too). XLA makes the cuFFT plans of an executable at its first run, on one
    host thread per GPU; cuFFT holds a process-wide lock while it loads each plan's CUDA module, and the load waits for its GPU to
    go idle, but that GPU runs a collective waiting for a peer whose host thread waits for the lock.

    XLA reads ``XLA_FLAGS`` at the first computation, so the flag is set here, on import, unless ``XLA_FLAGS`` already sets it
    (``--xla_gpu_use_memcpy_local_p2p=false`` turns it off). It changes no result, and does nothing on one GPU, on CPU, or with one
    GPU per process (``jax.distributed``), whose collectives are not local.
    """
    import os, sys
    flag, flags = '--xla_gpu_use_memcpy_local_p2p=true', os.environ.get('XLA_FLAGS', '')
    if 'xla_gpu_use_memcpy_local_p2p' in flags:
        return
    os.environ['XLA_FLAGS'] = ' '.join(filter(None, [flags, flag]))
    xla_bridge = sys.modules.get('jax._src.xla_bridge')
    if xla_bridge is not None and xla_bridge.backends_are_initialized():
        import jax, warnings
        if jax.local_device_count() > 1:
            warnings.warn(f'JAX computed before romanpower was imported, so XLA runs without {flag}: sharding a mesh over the '
                          f'{jax.local_device_count()} GPUs of this process may deadlock. Import romanpower before any JAX '
                          f'computation, or set XLA_FLAGS="{flag}" before starting Python.')


_set_xla_p2p_flag()

from .mesh import MeshAttrs, RealMeshField, ComplexMeshField, ParticleField, FKPField, r2c, c2r, apply, read, paint, fftfreq, create_sharding_mesh, create_sharded_array, create_sharded_random, exchange_particles, get_mesh_attrs, split_particles, compute_normalization, compute_box_normalization, get_wide_angle_selection_mesh
from .mesh2 import compute_mesh2, compute_mesh2_spectrum, compute_mesh2_correlation, BinMesh2SpectrumPoles, BinMesh2CorrelationPoles, compute_fkp2_normalization, compute_box2_normalization, compute_fkp2_shotnoise, compute_mesh2_spectrum_mean, compute_mesh2_spectrum_window, compute_smooth2_spectrum_window, get_smooth2_window_bin_attrs, interpolate_window_function
from .mesh3 import compute_mesh3, compute_mesh3_spectrum, BinMesh3SpectrumPoles, compute_mesh3_correlation, BinMesh3CorrelationPoles, compute_fkp3_normalization, compute_box3_normalization, compute_fkp3_shotnoise, get_smooth3_window_bin_attrs, compute_smooth3_spectrum_window, get_scoccimarro_symmetrization_matrix
from .meshn import compute_meshn_spectrum, compute_mesh4_spectrum, compute_mesh5_spectrum, compute_mesh6_spectrum, BinMeshNSpectrumPoles, compute_boxn_normalization
from .particle2 import compute_particle2, compute_particle2_shotnoise, BinParticle2CorrelationPoles, BinParticle2SpectrumPoles
from .angular import (AngularAttrs, AlmField, PixelField, to_alm, to_pixel, alm2map, compute_angular2, compute_angular2_spectrum, BinAngular2Spectrum,
BinAngular3Spectrum, compute_angular3_spectrum, compute_angular3_spectrum_window, compute_fkp_angular3_normalization, compute_fkp_angular3_shotnoise,
compute_angular_normalization, compute_fkp_angular2_normalization, compute_fkp_angular2_shotnoise, compute_angular2_spectrum_window, compute_angular2_spectrum_mean)
from .particle3 import compute_particle3, BinParticle3CorrelationPoles
from .cov2 import compute_fkp2_covariance_window, compute_mesh2_covariance_window, compute_spectrum2_covariance
# Only the entry point: cov3's own `compute_fkp2_covariance_window` is a different object from
# cov2's exported one above, and `compute_fkp3_covariance_window` is only useful together with it,
# so both stay behind `from romanpower.cov3 import ...`.
from .cov3 import compute_spectrum3_covariance
from .oqe import (SeparableWeighting, separable_inverse_weight, compute_oqe2_normalization, compute_oqe2_shotnoise, covariance_weight)
from .rotation import WindowRotationSpectrum2
from .mock import generate_gaussian_mesh, generate_spectrum2_mesh, generate_anisotropic_gaussian_mesh, generate_spectrum2_alm, generate_spectrum3_alm, generate_spectrum3_mesh, generate_uniform_particles
from .utils import setup_logging
from .types import (Mesh2SpectrumPole, Mesh2SpectrumPoles, Mesh2CorrelationPole, Mesh2CorrelationPoles,
Particle2SpectrumPole, Particle2SpectrumPoles, Particle2CorrelationPole, Particle2CorrelationPoles, Angular2Spectrum, Angular3Spectrum,
Mesh3SpectrumPoles, Mesh3SpectrumPoles, Mesh3CorrelationPoles, Mesh3CorrelationPoles, MeshNSpectrumPole, MeshNSpectrumPoles, CovarianceMatrix, WindowMatrix, read, write)