#include <omp.h>
static void inner_block(const float *restrict const f,
			float *restrict const fp,
			const int nx,
			const int ny,
			const float *restrict const model_padded2_dt2,
			const float *restrict const fd_coeff,
			const int bj,
			const int blocksize_x)
{

	int i;
	int j;
	float f_xx;
	const int x_start = bj * blocksize_x + 8;
	const int x_end = x_start + blocksize_x <= nx - 8 ?
	    x_start + blocksize_x : nx - 8;

	for (i = 8; i < ny - 8; i++) {
		for (j = x_start; j < x_end; j++) {
			f_xx =
			    (2 * fd_coeff[0] * f[i * nx + j] +
			     fd_coeff[1] *
			     (f[i * nx + j + 1] +
			      f[i * nx + j - 1] +
			      f[(i + 1) * nx + j] +
			      f[(i - 1) * nx + j]) +
			     fd_coeff[2] *
			     (f[i * nx + j + 2] +
			      f[i * nx + j - 2] +
			      f[(i + 2) * nx + j] +
			      f[(i - 2) * nx + j]) +
			     fd_coeff[3] *
			     (f[i * nx + j + 3] +
			      f[i * nx + j - 3] +
			      f[(i + 3) * nx + j] +
			      f[(i - 3) * nx + j]) +
			     fd_coeff[4] *
			     (f[i * nx + j + 4] +
			      f[i * nx + j - 4] +
			      f[(i + 4) * nx + j] +
			      f[(i - 4) * nx + j]) +
			     fd_coeff[5] *
			     (f[i * nx + j + 5] +
			      f[i * nx + j - 5] +
			      f[(i + 5) * nx + j] +
			      f[(i - 5) * nx + j]) +
			     fd_coeff[6] *
			     (f[i * nx + j + 6] +
			      f[i * nx + j - 6] +
			      f[(i + 6) * nx + j] +
			      f[(i - 6) * nx + j]) +
			     fd_coeff[7] *
			     (f[i * nx + j + 7] +
			      f[i * nx + j - 7] +
			      f[(i + 7) * nx + j] +
			      f[(i - 7) * nx + j]) +
			     fd_coeff[8] *
			     (f[i * nx + j + 8] +
			      f[i * nx + j - 8] +
			      f[(i + 8) * nx + j] + f[(i - 8) * nx + j]));

			fp[i * nx + j] =
			    (model_padded2_dt2[i * nx + j] * f_xx +
			     2 * f[i * nx + j] - fp[i * nx + j]);
		}
	}
}

static void inner(const float *restrict const f,
		  float *restrict const fp,
		  const int nx,
		  const int ny,
		  const float *restrict const model_padded2_dt2,
		  const float *restrict const sources,
		  const int *restrict const sources_x,
		  const int *restrict const sources_y,
		  const int num_sources, const int source_len,
		  const float *restrict const fd_coeff, const int step,
		  const int blocksize_x,
		  const int nbx)
{

	int i;
	int bj;
	int sx;
	int sy;

#pragma omp parallel for default(none) private(bi)
	for (bj = 0; bj < nbx; bj++) {
			inner_block(f, fp, nx, ny, model_padded2_dt2, fd_coeff,
				    bj, blocksize_x);
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
	const int blocksize_x = 32;
	const int nbx = (int)((float)(nx - 16) / blocksize_x) +
	    (int)(((nx - 16) % blocksize_x) != 0);

	for (step = 0; step < num_steps; step++) {
		inner(f, fp, nx, ny, model_padded2_dt2, sources, sources_x,
		      sources_y, num_sources, source_len, fd_coeff, step,
		      blocksize_x, nbx);

		tmp = f;
		f = fp;
		fp = tmp;
	}
}
