"""Test the propagators."""
import pytest
import numpy as np
from scipy.integrate import quad
from wave_2d_fd_perf.propagators import (VC1_gcc)

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
        if t>r:
            return 1/(2*np.pi)/np.sqrt(t**2-r**2) * f[int(t/dt)]*dt*dx**2*v
        else:
            return 0.0
    y = quad(g, 0, T)[0]
    return y

@pytest.fixture
def model_one():
    """Create a constant model, and the expected wavefield."""
    N = 10
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
    expected = np.array([green(x*dx, y*dx, sx*dx, sy*dx, dx, dt,
                                    (nsteps)*dt, 1500,
                                    source) for x in range(N) for y in range(N)])
    expected = expected.reshape([N, N])
    return {'model': model, 'dx': dx, 'dt': dt, 'nsteps': nsteps,
            'sources': np.array([source]), 'sx': np.array([sx]), 'sy': np.array([sy]),
            'expected': expected}

@pytest.fixture
def versions():
    """Return a list of implementations."""
    return [VC1_gcc]

def test_one_reflector(model_one, versions):
    """Verify that the numeric and analytic wavefields are similar."""

    for v in versions:
        _test_version(v, model_one, atol=1.5)

def _test_version(version, model, atol):
    """Run the test for one implementation."""
    v = version(model['model'], model['dx'], model['dt'])
    y = v.step(model['nsteps'], model['sources'], model['sx'], model['sy'])
    print('y', y.shape, 'exp', model['expected'].shape)
    assert np.allclose(y, model['expected'], atol=atol)
