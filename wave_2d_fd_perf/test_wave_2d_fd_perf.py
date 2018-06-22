"""Test the propagators."""
import pytest
import numpy as np
from scipy.integrate import quad
from wave_2d_fd_perf.propagators import (VC1_O2_gcc, VC1_O3_gcc, VC1_Ofast_gcc, VC2_O2_gcc, VC2_O3_gcc, VC2_Ofast_gcc, VC3_Ofast_gcc, VC3_Ofast_unroll_gcc, VC4_Ofast_gcc, VC4_Ofast_extra1_gcc, VC4_Ofast_extra2_gcc, VC4_Ofast_extra3_gcc, VC5_Ofast_gcc, VC6_Ofast_gcc, VC6_Ofast_256_gcc, VC7_Ofast_gcc, VC8_Ofast_gcc, VC9_Ofast_gcc, VC10_Ofast_gcc, VC11_Ofast_gcc, VC12_Ofast_gcc, VC13_Ofast_gcc, VC14_Ofast_gcc, VC15_Ofast_gcc, VF1_O2_gcc, VF1_O3_gcc, VF1_Ofast_gcc, VF2_Ofast_gcc, VF3_Ofast_gcc, VF4_Ofast_gcc, VF5_Ofast_gcc, VF6_Ofast_gcc, VF6_Ofast_autopar_gcc, VCython1, VCython2)
from wave_2d_fd_perf.vpytorch1 import VPytorch1
from wave_2d_fd_perf.vpytorch2 import VPytorch2
from wave_2d_fd_perf.vpytorch3 import VPytorch3

def ricker(freq, length, dt, peak_time):
    """Return a Ricker wavelet with the specified central frequency."""
    t = np.arange(-peak_time, (length)*dt - peak_time, dt, dtype=np.float32)
    y = ((1.0 - 2.0*(np.pi**2)*(freq**2)*(t**2))
         * np.exp(-(np.pi**2)*(freq**2)*(t**2)))
    return y

def green(x0, y0, x1, y1, dx, dt, T, v, f):
    """Use the 2D Green's function to determine the wavefield at a given
    location and time due to the given source.
    """
    r = np.sqrt(np.abs(x1-x0)**2 + np.abs(y1-y0)**2)/v
    def g(t):
        if t > r:
            return 1/(2*np.pi)/np.sqrt(t**2-r**2) * f[int(t/dt)]*dt*dx**2*v
        else:
            return 0.0
    y = quad(g, 0, T)[0]
    return y

@pytest.fixture
def model_one(N=10, calc_expected=True):
    """Create a constant model, and the expected wavefield."""
    model = np.ones([N, N], dtype=np.float32) * 1500
    dx = 5
    dt = 0.001
    sx = int(N/2)
    sy = sx
    # time is chosen to avoid reflections from boundaries
    T = 0.8*(sx * dx / 1500)
    nsteps = np.ceil(T/dt).astype(np.int)
    source = ricker(25, nsteps, dt, 0.05)

    # direct wave
    if calc_expected:
        expected = np.array([green(x*dx, y*dx, sx*dx, sy*dx, dx, dt,
                                   (nsteps)*dt, 1500,
                                   source) for x in range(N) for y in range(N)])
        expected = expected.reshape([N, N])
    else:
        expected = []
    return {'model': model, 'dx': dx, 'dt': dt, 'nsteps': nsteps,
            'sources': np.array([source]), 'sx': np.array([sx]), 'sy': np.array([sy]),
            'expected': expected}


@pytest.fixture
def model_two():
    """Create a random model and compare with VPy1 implementation."""
    N = 100
    np.random.seed(0)
    model = np.random.random([N, N]).astype(np.float32) * 3000 + 1500
    dx = 5
    dt = 0.0006
    nsteps = np.ceil(0.2/dt).astype(np.int)
    num_sources = 10
    sources_x = np.zeros(num_sources, dtype=np.int)
    sources_y = np.zeros(num_sources, dtype=np.int)
    sources = np.zeros([num_sources, nsteps], dtype=np.float32)
    for sourceIdx in range(num_sources):
        sources_x[sourceIdx] = np.random.randint(N)
        sources_y[sourceIdx] = np.random.randint(N)
        peak_time = np.round((0.05+np.random.rand()*0.05)/dt)*dt
        sources[sourceIdx, :] = ricker(25, nsteps, dt, peak_time)
    v = VC1_O2_gcc(model, dx, dt)
    expected = v.step(nsteps, sources, sources_x, sources_y)
    return {'model': model, 'dx': dx, 'dt': dt, 'nsteps': nsteps,
            'sources': sources, 'sx': sources_x, 'sy': sources_y,
            'expected': expected}


@pytest.fixture
def versions():
    """Return a list of implementations."""
    return [VC1_O2_gcc, VC1_O3_gcc, VC1_Ofast_gcc,
            VC2_O2_gcc, VC2_O3_gcc, VC2_Ofast_gcc,
            VC3_Ofast_gcc,
            VC3_Ofast_unroll_gcc,
            VC4_Ofast_gcc,
            VC4_Ofast_extra1_gcc,
            VC4_Ofast_extra2_gcc,
            VC4_Ofast_extra3_gcc,
            VC5_Ofast_gcc,
            VC6_Ofast_gcc,
            VC7_Ofast_gcc,
            VC8_Ofast_gcc,
            VC9_Ofast_gcc,
            VC10_Ofast_gcc,
            VC11_Ofast_gcc,
            VC12_Ofast_gcc,
            VC13_Ofast_gcc,
            VC14_Ofast_gcc,
            VC15_Ofast_gcc,
            VF1_O2_gcc,
            VF1_O3_gcc,
            VF1_Ofast_gcc,
            VF2_Ofast_gcc,
            VF3_Ofast_gcc,
            VF4_Ofast_gcc,
            VF5_Ofast_gcc,
            VF6_Ofast_gcc,
            VF6_Ofast_autopar_gcc,
            VCython1,
            VCython2,
            VPytorch1,
            VPytorch2,
            VPytorch3]


#def test_one_reflector(model_one, versions):
#    """Verify that the numeric and analytic wavefields are similar."""
#
#    for v in versions:
#        _test_version(v, model_one, atol=1.5)


def test_allclose(model_two, versions):
    """Verify that all implementations produce similar results."""

    for v in versions[1:]:
        print(v.__name__)
        _test_version(v, model_two, atol=5e-4)
        print(v.__name__, 'align 256')
        _test_version(v, model_two, atol=5e-4, align=256)


def _test_version(version, model, atol, align=None):
    """Run the test for one implementation."""
    v = version(model['model'], model['dx'], model['dt'], align=align)
    y = v.step(model['nsteps'], model['sources'], model['sx'], model['sy'])
    assert np.allclose(y, model['expected'], atol=atol)
