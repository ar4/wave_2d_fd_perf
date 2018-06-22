#!/usr/bin/env python
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('wave_2d_fd_perf', parent_package, top_path)
    config.add_subpackage('wave_2d_fd_perf')
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    from numpy.distutils.extension import Extension
    from Cython.Build import cythonize
    extensions = [Extension("wave_2d_fd_perf.vcython1", ["wave_2d_fd_perf/vcython1.pyx"],
                  extra_compile_args=['-march=native', '-Ofast']),
                  Extension("wave_2d_fd_perf/vcython2", ["wave_2d_fd_perf/vcython2.pyx"],
                  extra_compile_args=['-march=native', '-Ofast', '-fopenmp'],
                  extra_link_args=['-fopenmp'])]

    setup(configuration=configuration, ext_modules=cythonize(extensions))
#from numpy.distutils.core import setup
#setup(configuration=configuration)
