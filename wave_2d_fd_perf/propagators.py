"""Propagate a 2D wavefield using different implementations of a
finite difference method so that runtimes can be compared.
"""
from ctypes import c_int, c_float
import numpy as np
import wave_2d_fd_perf
from wave_2d_fd_perf import libvf1_O2_gcc, libvf1_O3_gcc, libvf1_Ofast_gcc, libvf2_Ofast_gcc, libvf3_Ofast_gcc, libvf4_Ofast_gcc, libvf5_Ofast_gcc, libvf6_Ofast_gcc

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


class VC(Propagator):
    """C implementations."""
    def __init__(self, libname, model, dx, dt=None):
        super(VC, self).__init__(model, dx, dt)

        self._libvc = np.ctypeslib.load_library(libname, wave_2d_fd_perf.__path__[0])
        self._libvc.step.argtypes = \
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
                 c_float,
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
        self._libvc.step(self.current_wavefield, self.previous_wavefield,
                         self.nx_padded, self.ny_padded, self.model_padded2_dt2,
                         self.dx,
                         sources, sources_x, sources_y, num_sources, source_len,
                         num_steps)

        if num_steps%2 != 0:
            tmp = self.current_wavefield
            self.current_wavefield = self.previous_wavefield
            self.previous_wavefield = tmp

        return self.current_wavefield[8:self.ny_padded-8, 8:self.nx_padded-8]


class VCa(VC):
    """C implementations with variable blocksize."""
    def __init__(self, libname, model, blocksize_y, blocksize_x, dx, dt=None):
        super(VCa, self).__init__(libname, model, dx, dt)
        self.blocksize_y = blocksize_y
        self.blocksize_x = blocksize_x

    def step(self, num_steps, sources=None, sources_x=None, sources_y=None):
        """Propagate wavefield."""

        num_sources = sources.shape[0]
        source_len = sources.shape[1]
        self._libvc.step(self.current_wavefield, self.previous_wavefield,
                         self.nx_padded, self.ny_padded, self.model_padded2_dt2,
                         self.dx,
                         sources, sources_x, sources_y, num_sources, source_len,
                         num_steps, self.blocksize_y, self.blocksize_x)

        if num_steps%2 != 0:
            tmp = self.current_wavefield
            self.current_wavefield = self.previous_wavefield
            self.previous_wavefield = tmp

        return self.current_wavefield[8:self.ny_padded-8, 8:self.nx_padded-8]


class VF(Propagator):
    """Fortran implementations."""
    def __init__(self, model, dx, dt=None):
        super(VF, self).__init__(model, dx, dt)

    def step(self, num_steps, sources=None, sources_x=None, sources_y=None):
        """Propagate wavefield."""

        self.fstep(self.current_wavefield.T, self.previous_wavefield.T,
                         self.model_padded2_dt2.T,
                         self.dx,
                         sources.T, sources_x, sources_y, num_steps)

        if num_steps%2 != 0:
            tmp = self.current_wavefield
            self.current_wavefield = self.previous_wavefield
            self.previous_wavefield = tmp

        return self.current_wavefield[8:self.ny_padded-8, 8:self.nx_padded-8]


class VC1_O2_gcc(VC):
    """A C implementation with loop in Laplacian."""
    def __init__(self, model, dx, dt=None):
        super(VC1_O2_gcc, self).__init__('libvc1_O2_gcc', model, dx, dt)


class VC1_O3_gcc(VC):
    """VC1 with -O3."""
    def __init__(self, model, dx, dt=None):
        super(VC1_O3_gcc, self).__init__('libvc1_O3_gcc', model, dx, dt)


class VC1_Ofast_gcc(VC):
    """VC1 with -Ofast."""
    def __init__(self, model, dx, dt=None):
        super(VC1_Ofast_gcc, self).__init__('libvc1_Ofast_gcc', model, dx, dt)


class VC2_O2_gcc(VC):
    """Like VC1 but with Laplacian loop manually unrolled."""
    def __init__(self, model, dx, dt=None):
        super(VC2_O2_gcc, self).__init__('libvc2_O2_gcc', model, dx, dt)


class VC2_O3_gcc(VC):
    """VC2 with -O3."""
    def __init__(self, model, dx, dt=None):
        super(VC2_O3_gcc, self).__init__('libvc2_O3_gcc', model, dx, dt)


class VC2_Ofast_gcc(VC):
    """VC2 with -Ofast."""
    def __init__(self, model, dx, dt=None):
        super(VC2_Ofast_gcc, self).__init__('libvc2_Ofast_gcc', model, dx, dt)


class VC3_Ofast_gcc(VC):
    """A vectorized version of VC1."""
    def __init__(self, model, dx, dt=None):
        super(VC3_Ofast_gcc, self).__init__('libvc3_Ofast_gcc', model, dx, dt)


class VC4_Ofast_gcc(VC):
    """A vectorized version of VC2."""
    def __init__(self, model, dx, dt=None):
        super(VC4_Ofast_gcc, self).__init__('libvc4_Ofast_gcc', model, dx, dt)


class VC5_Ofast_gcc(VC):
    """VC3 with x (inner) loop parallelized."""
    def __init__(self, model, dx, dt=None):
        super(VC5_Ofast_gcc, self).__init__('libvc5_Ofast_gcc', model, dx, dt)


class VC6_Ofast_gcc(VC):
    """VC3 with y (outer) loop parallelized."""
    def __init__(self, model, dx, dt=None):
        super(VC6_Ofast_gcc, self).__init__('libvc6_Ofast_gcc', model, dx, dt)


class VC7_Ofast_gcc(VC):
    """VC3 with collapsed x and y loops parallelized."""
    def __init__(self, model, dx, dt=None):
        super(VC7_Ofast_gcc, self).__init__('libvc7_Ofast_gcc', model, dx, dt)


class VC8_Ofast_gcc(VC):
    """VC6 with blocking, parallelized over y blocks."""
    def __init__(self, model, dx, dt=None):
        super(VC8_Ofast_gcc, self).__init__('libvc8_Ofast_gcc', model, dx, dt)


class VC9_Ofast_gcc(VC):
    """VC6 with blocking, parallelized over x blocks."""
    def __init__(self, model, dx, dt=None):
        super(VC9_Ofast_gcc, self).__init__('libvc9_Ofast_gcc', model, dx, dt)


class VC10_Ofast_gcc(VC):
    """VC6 with blocking, parallelized over all blocks."""
    def __init__(self, model, dx, dt=None):
        super(VC10_Ofast_gcc, self).__init__('libvc10_Ofast_gcc', model, dx, dt)
        

class VC8a_Ofast_gcc(VCa):
    """VC8, variable blocksize."""
    def __init__(self, model, blocksize_y, blocksize_x, dx, dt=None):
        super(VC8a_Ofast_gcc, self).__init__('libvc8a_Ofast_gcc', model, blocksize_y, blocksize_x, dx, dt)


class VC9a_Ofast_gcc(VCa):
    """VC9, variable blocksize."""
    def __init__(self, model, blocksize_y, blocksize_x, dx, dt=None):
        super(VC9a_Ofast_gcc, self).__init__('libvc9a_Ofast_gcc', model, blocksize_y, blocksize_x, dx, dt)


class VC10a_Ofast_gcc(VCa):
    """VC10, variable blocksize."""
    def __init__(self, model, blocksize_y, blocksize_x, dx, dt=None):
        super(VC10a_Ofast_gcc, self).__init__('libvc10a_Ofast_gcc', model, blocksize_y, blocksize_x, dx, dt)


class VF1_O2_gcc(VF):
    """A simple Fortran implementation."""
    def __init__(self, model, dx, dt=None):
        super(VF1_O2_gcc, self).__init__(model, dx, dt)
        self.fstep = libvf1_O2_gcc.vf1.step


class VF1_O3_gcc(VF):
    """VF1, with -O3."""
    def __init__(self, model, dx, dt=None):
        super(VF1_O3_gcc, self).__init__(model, dx, dt)
        self.fstep = libvf1_O3_gcc.vf1.step


class VF1_Ofast_gcc(VF):
    """VF1, with -Ofast."""
    def __init__(self, model, dx, dt=None):
        super(VF1_Ofast_gcc, self).__init__(model, dx, dt)
        self.fstep = libvf1_Ofast_gcc.vf1.step


class VF2_Ofast_gcc(VF):
    def __init__(self, model, dx, dt=None):
        super(VF2_Ofast_gcc, self).__init__(model, dx, dt)
        self.fstep = libvf2_Ofast_gcc.vf2.step


class VF3_Ofast_gcc(VF):
    """Like VF1, but uses forall."""
    def __init__(self, model, dx, dt=None):
        super(VF3_Ofast_gcc, self).__init__(model, dx, dt)
        self.fstep = libvf3_Ofast_gcc.vf3.step


class VF4_Ofast_gcc(VF):
    """Like VF3, but does not use pure function."""
    def __init__(self, model, dx, dt=None):
        super(VF4_Ofast_gcc, self).__init__(model, dx, dt)
        self.fstep = libvf4_Ofast_gcc.vf4.step


class VF5_Ofast_gcc(VF):
    """Like VF1, but uses do concurrent."""
    def __init__(self, model, dx, dt=None):
        super(VF5_Ofast_gcc, self).__init__(model, dx, dt)
        self.fstep = libvf5_Ofast_gcc.vf5.step


class VF6_Ofast_gcc(VF):
    """Like VF5, but does not use pure function."""
    def __init__(self, model, dx, dt=None):
        super(VF6_Ofast_gcc, self).__init__(model, dx, dt)
        self.fstep = libvf6_Ofast_gcc.vf6.step
