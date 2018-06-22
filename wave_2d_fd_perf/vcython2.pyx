cimport cython
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inner(const float [:, :] f,
           float [:, :] fp,
           const int nx,
           const int ny,
           const int nxi,
           const float [:, :] model_padded2_dt2,
           const float [:, :] sources,
           const int [:] sources_x,
           const int [:] sources_y,
           const int num_sources,
           const int source_len,
           const float [:] fd_coeff,
           const int step):
    
    cdef int i
    cdef int j
    cdef int sx
    cdef int sy
    cdef float f_xx
    
    for i in prange(8, ny - 8, nogil=True):
        for j in range(8, nxi + 8):
            f_xx = (fd_coeff[0] * f[i, j] +
                    fd_coeff[1] *
                    (f[i, j + 1] +
                     f[i, j - 1] +
                     f[i + 1, j] +
                     f[i - 1, j]) +
                    fd_coeff[2] *
                    (f[i, j + 2] +
                     f[i, j - 2] +
                     f[i + 2, j] +
                     f[i - 2, j]) +
                    fd_coeff[3] *
                    (f[i, j + 3] +
                     f[i, j - 3] +
                     f[i + 3, j] +
                     f[i - 3, j]) +
                    fd_coeff[4] *
                    (f[i, j + 4] +
                     f[i, j - 4] +
                     f[i + 4, j] +
                     f[i - 4, j]) +
                    fd_coeff[5] *
                    (f[i, j + 5] +
                     f[i, j - 5] +
                     f[i + 5, j] +
                     f[i - 5, j]) +
                    fd_coeff[6] *
                    (f[i, j + 6] +
                     f[i, j - 6] +
                     f[i + 6, j] +
                     f[i - 6, j]) +
                    fd_coeff[7] *
                    (f[i, j + 7] +
                     f[i, j - 7] +
                     f[i + 7, j] +
                     f[i - 7, j]) +
                    fd_coeff[8] *
                    (f[i, j + 8] +
                     f[i, j - 8] +
                     f[i + 8, j] +
                     f[i - 8, j]))

            fp[i, j] = (model_padded2_dt2[i, j] * f_xx +
                        2 * f[i, j] - fp[i, j])
            
    for i in range(num_sources):
        sx = sources_x[i] + 8;
        sy = sources_y[i] + 8;
        fp[sy, sx] += model_padded2_dt2[sy, sx] * sources[i, step]

@cython.boundscheck(False)
@cython.wraparound(False)
def step(float [:, :] f,
         float [:, :] fp,
         const int nx,
         const int ny,
         const int nxi,
         const float [:, :] model_padded2_dt2,
         const float [:, :] sources,
         const int [:] sources_x,
         const int [:] sources_y,
         const int num_sources,
         const int source_len,
         const int num_steps,
         const float [:] fd_coeff):

    cdef int step

    for step in range(num_steps):
        inner(f, fp, nx, ny, nxi, model_padded2_dt2, sources, sources_x,
              sources_y, num_sources, source_len, fd_coeff, step)

        f, fp = fp, f
