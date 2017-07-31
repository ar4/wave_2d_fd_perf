static void inner(const float *restrict const f,
		  float *restrict const fp,
	  	  float *restrict const f_xx,
		  const int nx,
		  const int ny,
	          const int nxi,
		  const float *restrict const model_padded2_dt2,
		  const float *restrict const sources,
		  const int *restrict const sources_x,
		  const int *restrict const sources_y,
		  const int num_sources, const int source_len,
		  const float *restrict const fd_coeff, const int step)
{

	int i;
	int j;
	int k;
	int sx;
	int sy;

	for (i = 8; i < ny - 8; i++) {
		for (j = 8; j < nxi + 8; j++) {
			f_xx[i * nx + j] = 2 * fd_coeff[0] * f[i * nx + j];
		}
	}

	for (k = 1; k < 9; k++) {
		for (i = 8; i < ny - 8; i++) {
			for (j = 8; j < nxi + 8; j++) {
				f_xx[i * nx + j] += fd_coeff[k] *
				    f[i * nx + j + k];
			}
		}
		for (i = 8; i < ny - 8; i++) {
			for (j = 8; j < nxi + 8; j++) {
				f_xx[i * nx + j] += fd_coeff[k] *
				     f[i * nx + j - k];
			}
		}
		for (i = 8; i < ny - 8; i++) {
			for (j = 8; j < nxi + 8; j++) {
				f_xx[i * nx + j] += fd_coeff[k] *
				     f[(i + k) * nx + j];
			}
		}
		for (i = 8; i < ny - 8; i++) {
			for (j = 8; j < nxi + 8; j++) {
				f_xx[i * nx + j] += fd_coeff[k] *
				     f[(i - k) * nx + j];
			}
		}
	}

	for (i = 8; i < ny - 8; i++) {
		for (j = 8; j < nxi + 8; j++) {

			fp[i * nx + j] =
			    (model_padded2_dt2[i * nx + j] * f_xx[i * nx + j] +
			     2 * f[i * nx + j] - fp[i * nx + j]);
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
	  float *restrict const f_xx,
	  const int nx,
	  const int ny,
	  const int nxi,
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

	for (step = 0; step < num_steps; step++) {
		inner(f, fp, f_xx, nx, ny, nxi, model_padded2_dt2, sources,
		      sources_x, sources_y, num_sources, source_len, fd_coeff,
		      step);

		tmp = f;
		f = fp;
		fp = tmp;
	}
}
