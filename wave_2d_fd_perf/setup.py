#!/usr/bin/env python
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('wave_2d_fd_perf',parent_package,top_path)
    config.add_extension(name='libvc1_gcc', sources=['vc1.c'], extra_compile_args=['-march=native', '-O3', '-std=c99'])
    config.add_extension(name='libvc2_gcc', sources=['vc2.c'], extra_compile_args=['-march=native', '-O3', '-std=c99', '-fopenmp'], extra_link_args=['-fopenmp'])
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration)
