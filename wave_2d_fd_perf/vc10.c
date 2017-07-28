#include <omp.h>
static void inner_block(const float *const restrict f,
			float *const restrict fp,
			const int nx,
			const int ny,
			const float *restrict const model_padded2_dt2,
			const float *restrict const fd_coeff,
			const int bi,
			const int bj,
			const int blocksize_y, const int blocksize_x)
{

	int i;
	int j;
	int k;
	float f_xx;
	const int y_start = bi * blocksize_y + 8;
	const int x_start = bj * blocksize_x + 8;
	const int y_end = y_start + blocksize_y <= ny - 8 ?
	    y_start + blocksize_y : ny - 8;
	const int x_end = x_start + blocksize_x <= nx - 8 ?
	    x_start + blocksize_x : nx - 8;

	for (i = y_start; i < y_end; i++) {
		for (j = x_start; j < x_end; j++) {
			f_xx = 2 * fd_coeff[0] * f[i * nx + j];
			for (k = 1; k < 9; k++) {
				f_xx += fd_coeff[k] *
				    (f[i * nx + j + k] +
				     f[i * nx + j - k] +
				     f[(i + k) * nx + j] + f[(i - k) * nx + j]);
			}

			fp[i * nx + j] =
			    (model_padded2_dt2[i * nx + j] * f_xx +
			     2 * f[i * nx + j] - fp[i * nx + j]);
		}
	}
}

static void inner(const float *const restrict f,
		  float *const restrict fp,
		  const int nx,
		  const int ny,
		  const float *restrict const model_padded2_dt2,
		  const float *restrict const sources,
		  const int *restrict const sources_x,
		  const int *restrict const sources_y,
		  const int num_sources, const int source_len,
		  const float *restrict const fd_coeff, const int step,
		  const int blocksize_y, const int blocksize_x,
		  const int nby, const int nbx)
{

	int i;
	int bi;
	int bj;
	int sx;
	int sy;

#pragma omp parallel for default(none) collapse(2)
	for (bi = 0; bi < nby; bi++) {
		for (bj = 0; bj < nbx; bj++) {
			inner_block(f, fp, nx, ny, model_padded2_dt2, fd_coeff,
				    bi, bj, blocksize_y, blocksize_x);
		}
	}

	for (i = 0; i < num_sources; i++) {
		sx = sources_x[i] + 8;
		sy = sources_y[i] + 8;
		fp[sy * nx + sx] +=
		    (model_padded2_dt2[sy * nx + sx] *
		     sources[i * source_len + step]);
	}

}

void step(float *restrict f,
	  float *restrict fp,
	  const int nx,
	  const int ny,
	  const float *restrict const model_padded2_dt2,
	  const float dx,
	  const float *restrict const sources,
	  const int *restrict const sources_x,
	  const int *restrict const sources_y,
	  const int num_sources, const int source_len, const int num_steps)
{

	int step;
	float *tmp;
	float fd_coeff[9] = {
		-924708642.0f / 302702400 / (dx * dx),
		538137600.0f / 302702400 / (dx * dx),
		-94174080.0f / 302702400 / (dx * dx),
		22830080.0f / 302702400 / (dx * dx),
		-5350800.0f / 302702400 / (dx * dx),
		1053696.0f / 302702400 / (dx * dx),
		-156800.0f / 302702400 / (dx * dx),
		15360.0f / 302702400 / (dx * dx),
		-735.0f / 302702400 / (dx * dx)
	};
	const int blocksize_y = 8;
	const int blocksize_x = 32;
	const int nby = (int)((float)(ny - 16) / blocksize_y) +
	    (int)(((ny - 16) % blocksize_y) != 0);
	const int nbx = (int)((float)(nx - 16) / blocksize_x) +
	    (int)(((nx - 16) % blocksize_x) != 0);

	for (step = 0; step < num_steps; step++) {
		inner(f, fp, nx, ny, model_padded2_dt2, sources, sources_x,
		      sources_y, num_sources, source_len, fd_coeff, step,
		      blocksize_y, blocksize_x, nby, nbx);

		tmp = f;
		f = fp;
		fp = tmp;
	}
}
