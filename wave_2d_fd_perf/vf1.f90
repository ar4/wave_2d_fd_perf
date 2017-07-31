module vf1

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
                integer :: k
                integer :: sx
                integer :: sy
                integer :: ny
                integer :: num_sources
                real :: f_xx

                ny = size(f, dim=2)
                num_sources = size(sources, dim=2)

                do i = 9, ny - 8
                do j = 9, nxi + 8
                f_xx = 2 * fd_coeff(1) * f(j, i)
                do k = 1, 8
                f_xx = f_xx + fd_coeff(k + 1) *                        &
                        (f(j, i + k) + f(j, i - k) +                   &
                        f(j + k, i) + f(j - k, i))
                end do
                fp(j, i) = (model_padded2_dt2(j, i) * f_xx +           &
                        2 * f(j, i) - fp(j, i))
                end do
                end do

                do i = 1, num_sources
                sx = sources_x(i) + 9
                sy = sources_y(i) + 9
                fp(sx, sy) = fp(sx, sy) + (model_padded2_dt2(sx, sy)   &
                        * sources(step_idx, i))
                end do

        end subroutine step_inner

end module vf1
