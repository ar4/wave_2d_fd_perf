"""Propagate a 2D wavefield using different implementations of a
finite difference method so that runtimes can be compared.
"""
from ctypes import c_int, c_float
import numpy as np
import wave_2d_fd_perf

class Propagator(object):
    """A finite difference propagator for the 2D wave equation."""
    def __init__(self, model, dx, dt=None):
        self.nx = model.shape[1]
        self.ny = model.shape[0]
        self.dx = np.float32(dx)
        max_vel = np.max(model)
        if dt:
            self.dt = dt
        else:
            self.dt = 0.6 * self.dx / max_vel
        self.nx_padded = self.nx + 2*8
        self.ny_padded = self.ny + 2*8
        self.model_padded = np.pad(model, ((8, 8), (8, 8)), 'edge')
        self.model_padded2_dt2 = self.model_padded**2 * self.dt**2
        self.wavefield = [np.zeros([self.ny_padded, self.nx_padded], np.float32),
                          np.zeros([self.ny_padded, self.nx_padded], np.float32)
                         ]
        self.current_wavefield = self.wavefield[0]
        self.previous_wavefield = self.wavefield[1]


class VC1_gcc(Propagator):
    """A C implementation."""
    def __init__(self, model, dx, dt=None):
        super(VC1_gcc, self).__init__(model, dx, dt)

        self._libvc1 = np.ctypeslib.load_library('libvc1_gcc', wave_2d_fd_perf.__path__[0])
        self._libvc1.step.argtypes = \
                [np.ctypeslib.ndpointer(dtype=np.float32, ndim=2,
                                        shape=(self.ny_padded, self.nx_padded),
                                        flags=('C_CONTIGUOUS', 'WRITEABLE')),
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=2,
                                        shape=(self.ny_padded, self.nx_padded),
                                        flags=('C_CONTIGUOUS', 'WRITEABLE')),
                 c_int, c_int,
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=2,
                                        shape=(self.ny_padded, self.nx_padded),
                                        flags=('C_CONTIGUOUS')),
                 c_float, c_float,
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=2,
                                        flags=('C_CONTIGUOUS')),
                 np.ctypeslib.ndpointer(dtype=np.int, ndim=1,
                                        flags=('C_CONTIGUOUS')),
                 np.ctypeslib.ndpointer(dtype=np.int, ndim=1,
                                        flags=('C_CONTIGUOUS')),
                 c_int, c_int, c_int]

    def step(self, num_steps, sources=None, sources_x=None, sources_y=None):
        """Propagate wavefield."""

        num_sources = sources.shape[0]
        source_len = sources.shape[1]
        self._libvc1.step(self.current_wavefield, self.previous_wavefield,
                          self.nx_padded, self.ny_padded, self.model_padded,
                          self.dt, self.dx,
                          sources, sources_x, sources_y, num_sources, source_len,
                          num_steps)

        if num_steps%2 != 0:
            tmp = self.current_wavefield
            self.current_wavefield = self.previous_wavefield
            self.previous_wavefield = tmp

        return self.current_wavefield[8:self.ny_padded-8, 8:self.nx_padded-8]
