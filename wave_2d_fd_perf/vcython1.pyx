cimport cython
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
    cdef int i
    cdef int j
    cdef int k
    cdef int sx
    cdef int sy
    cdef float f_xx

    for step in range(num_steps):
        for i in range(8, ny - 8):
            for j in range(8, nxi + 8):
                f_xx = fd_coeff[0] * f[i, j]
                for k in range(1, 9):
                    f_xx += (fd_coeff[k] *
                             (f[i, j + k] +
                              f[i, j - k] +
                              f[(i + k), j] +
                              f[(i - k), j]))

                fp[i, j] = (model_padded2_dt2[i, j] * f_xx +
                            2 * f[i, j] - fp[i, j])

        for i in range(num_sources):
            sx = sources_x[i] + 8;
            sy = sources_y[i] + 8;
            fp[sy, sx] += model_padded2_dt2[sy, sx] * sources[i, step]

        f, fp = fp, f
