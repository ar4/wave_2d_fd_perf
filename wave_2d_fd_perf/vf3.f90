module vf3

        implicit none

contains

        subroutine step(f1, f2, model_padded2_dt2, nxi, dx, sources,   &
                        sources_x, sources_y, num_steps)

                real, intent (in out), dimension (:, :) :: f1
                real, intent (in out), dimension (:, :) :: f2
                real, intent (in), dimension (:, :) :: model_padded2_dt2
                integer, intent (in) :: nxi
                real, intent (in) :: dx
                real, intent (in), dimension (:, :) :: sources
                integer, intent (in), dimension (:) :: sources_x
                integer, intent (in), dimension (:) :: sources_y
                integer, intent (in) :: num_steps

                integer :: step_idx
                logical :: even
                real, dimension(9) :: fd_coeff

                fd_coeff = (/                                          &
                        -924708642.0 / 302702400 / (dx * dx),          &
                        538137600.0 / 302702400 / (dx * dx),           &
                        -94174080.0 / 302702400 / (dx * dx),           &
                        22830080.0 / 302702400 / (dx * dx),            &
                        -5350800.0 / 302702400 / (dx * dx),            &
                        1053696.0 / 302702400 / (dx * dx),             &
                        -156800.0 / 302702400 / (dx * dx),             &
                        15360.0 / 302702400 / (dx * dx),               &
                        -735.0 / 302702400 / (dx * dx)                 &
                        /)


                do step_idx = 1, num_steps
                even = (mod (step_idx, 2) == 0)
                if (even) then
                        call step_inner(f2, f1, model_padded2_dt2, nxi,&
                                sources, sources_x, sources_y,         &
                                step_idx, fd_coeff)
                else
                        call step_inner(f1, f2, model_padded2_dt2, nxi,&
                                sources, sources_x, sources_y,         &
                                step_idx, fd_coeff)
                end if
                end do

        end subroutine step


        subroutine step_inner(f, fp, model_padded2_dt2, nxi, sources,  &
                        sources_x, sources_y, step_idx, fd_coeff)

                real, intent (in), dimension (:, :) :: f
                real, intent (in out), dimension (:, :) :: fp
                real, intent (in), dimension (:, :) :: model_padded2_dt2
                integer, intent (in) :: nxi
                real, intent (in), dimension (:, :) :: sources
                integer, intent (in), dimension (:) :: sources_x
                integer, intent (in), dimension (:) :: sources_y
                integer, intent (in) :: step_idx
                real, intent (in), dimension (9) :: fd_coeff

                integer :: i
                integer :: j
                integer :: ny
                integer :: num_sources

                ny = size(f, dim=2)
                num_sources = size(sources, dim=2)

                forall (i = 9 : ny - 8, j = 9 : nxi + 8)
                        fp(j, i) = step_update(f, fp(j, i),            &
                                model_padded2_dt2(j, i), i, j, fd_coeff)
                end forall

                forall (i = 1 : num_sources)
                        fp(sources_x(i) + 9, sources_y(i) + 9) =       &
                                add_source(fp(sources_x(i) + 9,        &
                                sources_y(i) + 9),                     &
                                model_padded2_dt2(sources_x(i) + 9,    &
                                sources_y(i) + 9),                     &
                                sources(step_idx, i))
                end forall

        end subroutine step_inner


        pure function step_update(f, fp, model_padded2_dt2, i, j,      &
                        fd_coeff)

                real, intent (in), dimension (:, :) :: f
                real, intent (in) :: fp
                real, intent (in) :: model_padded2_dt2
                integer, intent (in) :: i
                integer, intent (in) :: j
                real, intent (in), dimension (9) :: fd_coeff

                real :: step_update

                step_update = (model_padded2_dt2 *                     &
                        laplacian(f, i, j, fd_coeff) + 2 * f(j, i) - fp)

        end function step_update


        pure function laplacian(f, i, j, fd_coeff)

                real, intent (in), dimension (:, :) :: f
                integer, intent (in) :: i
                integer, intent (in) :: j
                real, intent (in), dimension (9) :: fd_coeff

                real :: laplacian
                integer :: k

                laplacian = 2 * fd_coeff(1) * f(j, i)
                do k = 1, 8
                laplacian = laplacian + fd_coeff(k + 1) *              &
                        (f(j, i + k) + f(j, i - k) +                   &
                        f(j + k, i) + f(j - k, i))
                end do

        end function laplacian


        pure function add_source(fp, model_padded2_dt2, source)

                real, intent (in) :: fp
                real, intent (in) :: model_padded2_dt2
                real, intent (in) :: source

                real :: add_source

                add_source = fp + model_padded2_dt2 * source

        end function add_source

end module vf3
