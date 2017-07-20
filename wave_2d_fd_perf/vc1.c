void step(float *restrict f,
	  float *restrict fp,
	  const int nx,
	  const int ny,
	  const float *restrict const model_padded,
	  const float dt,
	  const float dx,
	  const float *restrict const sources,
	  const int *restrict const sources_x,
	  const int *restrict const sources_y,
	  const int num_sources, const int source_len, const int num_steps)
{

	int step;
	int i;
	int j;
	int sx;
	int sy;
	float f_xx;
	float *tmp;
	float fd_coeff[9] = {
		-924708642.0/302702400,
		538137600.0/302702400,
		-94174080.0/302702400,
		22830080.0/302702400,
		-5350800.0/302702400,
		1053696.0/302702400,
		-156800.0/302702400,
		15360.0/302702400,
		-735.0/302702400};

	for (step = 0; step < num_steps; step++) {
		for (i = 8; i < ny - 8; i++) {
			for (j = 8; j < nx - 8; j++) {
				f_xx =
					(2 * fd_coeff[0] * f[i * nx + j] +
					 fd_coeff[1] * (f[i * nx + j + 1] + f[i * nx + j - 1] +
						 f[(i + 1) * nx + j] + f[(i - 1) * nx + j]) + 
					 fd_coeff[2] * (f[i * nx + j + 2] + f[i * nx + j - 2] +
						 f[(i + 2) * nx + j] + f[(i - 2) * nx + j]) + 
					 fd_coeff[3] * (f[i * nx + j + 3] + f[i * nx + j - 3] +
						 f[(i + 3) * nx + j] + f[(i - 3) * nx + j]) + 
					 fd_coeff[4] * (f[i * nx + j + 4] + f[i * nx + j - 4] +
						 f[(i + 4) * nx + j] + f[(i - 4) * nx + j]) + 
					 fd_coeff[5] * (f[i * nx + j + 5] + f[i * nx + j - 5] +
						 f[(i + 5) * nx + j] + f[(i - 5) * nx + j]) + 
					 fd_coeff[6] * (f[i * nx + j + 6] + f[i * nx + j - 6] +
						 f[(i + 6) * nx + j] + f[(i - 6) * nx + j]) + 
					 fd_coeff[7] * (f[i * nx + j + 7] + f[i * nx + j - 7] +
						 f[(i + 7) * nx + j] + f[(i - 7) * nx + j]) + 
					 fd_coeff[8] * (f[i * nx + j + 8] + f[i * nx + j - 8] +
						 f[(i + 8) * nx + j] + f[(i - 8) * nx + j])) /
					(dx * dx);

				fp[i * nx + j] =
				    (model_padded[i * nx + j] *
				     model_padded[i * nx + j] * dt * dt * f_xx +
				     2 * f[i * nx + j] - fp[i * nx + j]);
			}
		}

		for (i = 0; i < num_sources; i++) {
			sx = sources_x[i] + 8;
			sy = sources_y[i] + 8;
			fp[sy * nx + sx] +=
			    (model_padded[sy * nx + sx] *
			     model_padded[sy * nx + sx] *
			     dt * dt * sources[i * source_len + step]);
		}

		tmp = f;
		f = fp;
		fp = tmp;
	}
}
