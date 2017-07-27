#!/usr/bin/env python
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('wave_2d_fd_perf',parent_package,top_path)
    config.add_extension(name='libvc1_O2_gcc', sources=['vc1.c'], extra_compile_args=['-Wall', '-Wextra', '-pedantic', '-Werror', '-march=native', '-O2', '-std=c11'])
    config.add_extension(name='libvc1_O3_gcc', sources=['vc1.c'], extra_compile_args=['-Wall', '-Wextra', '-pedantic', '-Werror', '-march=native', '-O3', '-std=c11'])
    config.add_extension(name='libvc1_Ofast_gcc', sources=['vc1.c'], extra_compile_args=['-Wall', '-Wextra', '-pedantic', '-Werror', '-march=native', '-Ofast', '-std=c11'])
    config.add_extension(name='libvc2_O2_gcc', sources=['vc2.c'], extra_compile_args=['-Wall', '-Wextra', '-pedantic', '-Werror', '-march=native', '-O2', '-std=c11'])
    config.add_extension(name='libvc2_O3_gcc', sources=['vc2.c'], extra_compile_args=['-Wall', '-Wextra', '-pedantic', '-Werror', '-march=native', '-O3', '-std=c11'])
    config.add_extension(name='libvc2_Ofast_gcc', sources=['vc2.c'], extra_compile_args=['-Wall', '-Wextra', '-pedantic', '-Werror', '-march=native', '-Ofast', '-std=c11'])
    config.add_extension(name='libvc3_Ofast_gcc', sources=['vc3.c'], extra_compile_args=['-Wall', '-Wextra', '-pedantic', '-Werror', '-march=native', '-Ofast', '-std=c11'])
    config.add_extension(name='libvc4_Ofast_gcc', sources=['vc4.c'], extra_compile_args=['-Wall', '-Wextra', '-pedantic', '-Werror', '-march=native', '-Ofast', '-std=c11'])
    config.add_extension(name='libvc5_Ofast_gcc', sources=['vc5.c'], extra_compile_args=['-Wall', '-Wextra', '-pedantic', '-Werror', '-march=native', '-Ofast', '-std=c11', '-fopenmp'], extra_link_args=['-fopenmp'])
    config.add_extension(name='libvc6_Ofast_gcc', sources=['vc6.c'], extra_compile_args=['-Wall', '-Wextra', '-pedantic', '-Werror', '-march=native', '-Ofast', '-std=c11', '-fopenmp'], extra_link_args=['-fopenmp'])
    config.add_extension(name='libvc7_Ofast_gcc', sources=['vc7.c'], extra_compile_args=['-Wall', '-Wextra', '-pedantic', '-Werror', '-march=native', '-Ofast', '-std=c11', '-fopenmp'], extra_link_args=['-fopenmp'])
    config.add_extension(name='libvc8_Ofast_gcc', sources=['vc8.c'], extra_compile_args=['-Wall', '-Wextra', '-pedantic', '-Werror', '-march=native', '-Ofast', '-std=c11', '-fopenmp'], extra_link_args=['-fopenmp'])
    config.add_extension(name='libvc9_Ofast_gcc', sources=['vc9.c'], extra_compile_args=['-Wall', '-Wextra', '-pedantic', '-Werror', '-march=native', '-Ofast', '-std=c11', '-fopenmp'], extra_link_args=['-fopenmp'])
    config.add_extension(name='libvc10_Ofast_gcc', sources=['vc10.c'], extra_compile_args=['-Wall', '-Wextra', '-pedantic', '-Werror', '-march=native', '-Ofast', '-std=c11', '-fopenmp'], extra_link_args=['-fopenmp'])
    config.add_extension(name='libvc8a_Ofast_gcc', sources=['vc8a.c'], extra_compile_args=['-Wall', '-Wextra', '-pedantic', '-Werror', '-march=native', '-Ofast', '-std=c11', '-fopenmp'], extra_link_args=['-fopenmp'])
    config.add_extension(name='libvc9a_Ofast_gcc', sources=['vc9a.c'], extra_compile_args=['-Wall', '-Wextra', '-pedantic', '-Werror', '-march=native', '-Ofast', '-std=c11', '-fopenmp'], extra_link_args=['-fopenmp'])
    config.add_extension(name='libvc10a_Ofast_gcc', sources=['vc10a.c'], extra_compile_args=['-Wall', '-Wextra', '-pedantic', '-Werror', '-march=native', '-Ofast', '-std=c11', '-fopenmp'], extra_link_args=['-fopenmp'])
    config.add_extension(name='libvf1_O2_gcc', sources=['vf1.f90'], extra_f90_compile_args=['-Wall', '-Wextra', '-pedantic', '-Werror', '-march=native', '-O2', '-std=f95'])

    #config.add_extension(name='libvc2_gcc', sources=['vc2.c'], extra_compile_args=['-Wall', '-Wextra', '-pedantic', '-Werror', '-march=native', '-O3', '-std=c11', '-fopenmp'], extra_link_args=['-Wall', '-Wextra', '-pedantic', '-Werror''-fopenmp'])
    #config.add_extension(name='libvc3_gcc', sources=['vc3.c'], extra_compile_args=['-Wall', '-Wextra', '-pedantic', '-Werror', '-march=native', '-O3', '-std=c11', '-fopenmp'], extra_link_args=['-Wall', '-Wextra', '-pedantic', '-Werror''-fopenmp'])
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration)
