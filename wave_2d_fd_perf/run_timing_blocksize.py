"""Measure the runtime of the propagators."""
from timeit import repeat
import numpy as np
import pandas as pd
from wave_2d_fd_perf.propagators import (VC8a_Ofast_gcc, VC9a_Ofast_gcc, VC10a_Ofast_gcc)
from wave_2d_fd_perf.test_wave_2d_fd_perf import ricker

def run_timing_model_size(num_repeat=10, num_steps=10, model_sizes=range(200, 1200, 200), versions=None, blocksizes_y=None, blocksizes_x=None, align=None):
    """Time implementations as model size varies."""

    if versions == None:
        versions = _versions()

    if blocksizes_y == None:
        blocksizes_y = _blocksizes()

    if blocksizes_x == None:
        blocksizes_x = _blocksizes()

    times = pd.DataFrame(columns=['version', 'blocksize_y', 'blocksize_x', 'num_steps', 'model_size', 'time'])

    for N in model_sizes:
        model = _make_model(N, num_steps)
        times = _time_versions(versions, blocksizes_y, blocksizes_x, model, num_repeat, times, align)

    return times


def _versions():
    """Return a list of versions to be timed."""
    return [{'class': VC8a_Ofast_gcc, 'name': 'C v8a (gcc, -Ofast)'},
            {'class': VC9a_Ofast_gcc, 'name': 'C v9a (gcc, -Ofast)'},
            {'class': VC10a_Ofast_gcc, 'name': 'C v10a (gcc, -Ofast)'}]


def _blocksizes():
    """Return a list of blocksizes to try."""
    return [8, 16, 32, 64, 128]


def _make_model(N, nsteps):
    """Create a model with a given number of elements and time steps."""
    model = np.random.random([N, N]).astype(np.float32) * 3000 + 1500
    max_vel = 4500
    dx = 5
    dt = 0.001
    source = ricker(25, nsteps, dt, 0.05)
    sx = int(N/2)
    sy = sx
    return {'model': model, 'dx': dx, 'dt': dt, 'nsteps': nsteps,
            'sources': np.array([source]), 'sx': np.array([sx]),
            'sy': np.array([sy])}


def _time_versions(versions, blocksizes_y, blocksizes_x, model, num_repeat, dataframe, align=None):
    """Loop over versions and append the timing results to the dataframe."""
    num_steps = model['nsteps']
    model_size = len(model['model'])
    for v in versions:
        for blocksize_y in blocksizes_y:
            for blocksize_x in blocksizes_x:
                time = _time_version(v['class'], model,
                                     blocksize_y, blocksize_x, num_repeat, align)
                dataframe = dataframe.append({'version': v['name'],
                                              'blocksize_y': blocksize_y,
                                              'blocksize_x': blocksize_x,
                                              'num_steps': num_steps,
                                              'model_size': model_size,
                                              'time': time}, ignore_index=True)
    return dataframe


def _time_version(version, model, blocksize_y, blocksize_x, num_repeat, align=None):
    """Time a particular version."""
    v = version(model['model'], blocksize_y, blocksize_x, model['dx'], model['dt'], align)

    def closure():
        """Closure over variables so they can be used in repeat below."""
        v.step(model['nsteps'], model['sources'], model['sx'], model['sy'])
 
    return np.min(repeat(closure, number=1, repeat=num_repeat))

if __name__ == '__main__':
    print(run_timing_num_steps())
