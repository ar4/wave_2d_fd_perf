#include <omp.h>
static void inner(const float *restrict const f,
		  float *restrict const fp,
		  const int nx,
		  const float *restrict const model_padded2_dt2,
		  const float *restrict const sources,
		  const int *restrict const sources_x,
		  const int *restrict const sources_y,
		  const int source_len,
		  const float *restrict const fd_coeff, const int step,
		  const int thread_start, const int thread_end,
		  const int thread_source_start, const int thread_source_end)
{

	int i;
	int j;
	int sx;
	int sy;
	float f_xx;

	for (i = thread_start; i < thread_end; i++) {
		for (j = 8; j < nx - 8; j++) {
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

	for (i = thread_source_start; i < thread_source_end; i++) {
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
	const float fd_coeff[9] = {
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
	int threadIdx;
	int thread_start;
	int thread_end;
	int thread_source_start;
	int thread_source_end;
	int per_thread;
	int i;

#pragma omp parallel default(none) private(thread_start, thread_end, \
		thread_source_start, thread_source_end, step, tmp, \
		threadIdx, per_thread, i) firstprivate(f, fp)
	{
		per_thread = (int)((float)(ny - 16) / omp_get_num_threads()) +
		    (int)(((ny - 16) % omp_get_num_threads()) != 0);
		threadIdx = omp_get_thread_num();

		thread_start = 8 + per_thread * threadIdx;
		thread_end = thread_start + per_thread;
		thread_end = thread_end < ny - 8 ? thread_end : ny - 8;

		thread_source_start = -1;
		thread_source_end = -1;
		// Find the first source index that is within this thread's range
		for (i = 0; i < num_sources; i++) {
			if (sources_y[i] + 8 < thread_start)
				continue;
			if (sources_y[i] + 8 > thread_end)
				break;
			thread_source_start = i;
			thread_source_end = i + 1;
			break;
		}

		// Find the last source index that is within this thread's range
		if (thread_source_end >= 0) {
			for (i = thread_source_end; i < num_sources; i++) {
				if (sources_y[i] + 8 > thread_end)
					break;
				thread_source_end = i + 1;
			}
		}

		for (step = 0; step < num_steps; step++) {
			inner(f, fp, nx, model_padded2_dt2, sources, sources_x,
			      sources_y, source_len, fd_coeff, step,
			      thread_start, thread_end,
			      thread_source_start, thread_source_end);

			tmp = f;
			f = fp;
			fp = tmp;
#pragma omp barrier
		}
	}
}
