!---------------------------------------------------------------------------------
!       Fortran utility to do BAO analysis
!       credit Zhejie Ding, log_prob_nonlinear_BAO.f95
!       https://github.com/zdplayground/emcee_fit_Pwnw
!
!       compile with > f2py baoutils.f95 -m baoutils -h baoutils.pyf --overwrite-signature
!       build with > f2py -c baoutils.pyf baoutils.f95
!
!
!       
! 
! 
!
subroutine match_params(theta, params_indices, fix_params, params_array, dim_theta, dim_params)

    ! match params 
    implicit none
    
    ! input
    integer, intent(in)                         :: dim_theta, dim_params
    
    real(8), dimension(dim_theta), intent(in)   :: theta
    real(8), dimension(dim_params), intent(in)  :: params_indices, fix_params
    
    ! output
    real(8), dimension(dim_params), intent(out) :: params_array
    
    ! local variables
    integer :: counter, i

    ! Be very careful that the starting value
    ! is different from Python's. It's 1 in fortran!
    counter = 1  
    do i=1, dim_params
        if (params_indices(i) == 1) then
            params_array(i) = theta(counter)
            counter = counter + 1
        else
            params_array(i) = fix_params(i)
        end if
    end do
    return
end subroutine match_params



subroutine cal_Pk_model_prebeutler(Pk_linw, Pk_sm, k_t, a1, a2, a3, a4, a5, b1, sigmas2, sigmanl2, Pk_model, dim_kt)
    ! Beutler et. al. 2017 (modified)
    ! P = [A + B Psm] [1 + (Plin/Psm-1) C]
    !
    ! A = a1k + a2 + a3/k + a4k^2 + a5k^3      Polynomial 
    ! B = B x exp[1 + 0.5 Sigma_s^2 k^2]^-1    Finger of God
    ! C = exp [-0.5*k^2 Sigma_nl^2] 
    ! 
    ! 
    implicit none

    ! input
    integer, intent(in)                     :: dim_kt
    real(8), intent(in)                     :: a1, a2, a3, a4, a5, b1, sigmas2, sigmanl2
    real(8), dimension(dim_kt), intent(in)  :: Pk_linw, Pk_sm, k_t
    
    ! output
    real(8), dimension(dim_kt), intent(out) :: Pk_model
    
    ! local
    integer :: i
    real(8) :: k, k2, k3, A, B, C
    
    do i=1, dim_kt
        !
        k  = k_t(i)
        k2 = k_t(i)*k_t(i)
        k3 = k2 * k_t(i)
        
        
        A  = a1*k + a2 + a3/k + a4*k2 + a5*k3
        B  = b1/exp(1.0d0 + 0.5d0*k2*sigmas2)
        C  = exp(-0.5d0*k2*sigmanl2)
        
        
        Pk_model(i) = (A + B*Pk_sm(i)) * (1.0d0 + (Pk_linw(i)/Pk_sm(i) - 1.0d0)*C)
        
    end do
    
    return
end subroutine cal_Pk_model_prebeutler

subroutine lnprior_prebeutler(theta, params_indices, fix_params, lp, dim_theta, dim_params)

    ! lnPrior    
    implicit none
    
    integer, intent(in)                       :: dim_theta, dim_params    
    real(8), dimension(dim_theta), intent(in)  :: theta
    real(8), dimension(dim_params), intent(in) :: params_indices, fix_params
        
    ! output
    real(8), intent(out) :: lp
    
    ! local    
    real(8), dimension(dim_params) :: params_array    
    real(8) :: alpha, a1, a2, a3, a4, a5, b1, sigmas2, sigmanl2
    
    ! match params
    call match_params(theta, params_indices, fix_params, params_array, dim_theta, dim_params)
    
    
    alpha  = params_array(1)
    a1     = params_array(2)
    a2     = params_array(3)
    a3     = params_array(4)
    a4     = params_array(5)
    a5     = params_array(6)
    b1      = params_array(7)
    sigmas2 = params_array(8)  
    sigmanl2 = params_array(9)
   !  1.00303509e+00  1.07103826e+04 -1.91184682e+03 -1.82141962e+01
 !-1.59015631e+04 -2.07271427e+04  2.76553040e+00  5.69430688e+01        
    if (alpha > 0.8d0 .and. alpha<1.2d0 &
        .and. a1 > -3.d4 .and. a1 < 3.d4 &
        .and. a2 > -3.d4 .and. a2 < 3.d4 &
        .and. a3 > -3.d2 .and. a3 < 3.d2 &
        .and. a4 > -3.d4 .and. a4 < 3.d4 &
        .and. a5 > -3.d4 .and. a5 < 3.d4 &
        .and. b1 > -10d0 .and. b1 < 10d0 &        
        .and. sigmas2>-0.1d0 .and. sigmas2<200.d0 &
        .and. sigmanl2>-0.1d0 .and. sigmanl2 < 200.d0) then
        lp = 0.d0
    else
        lp = -1.d30  ! return a negative infinitely large number
    endif
    !print*, lp
    return
end subroutine lnprior_prebeutler



subroutine cal_Pk_model(Pk_linw, Pk_sm, k_t, sigma2, A, B, Pk_model, dim_kt)

    ! Pwig(k') = A*((Plin(k) - Psm(k))*exp(-k^2*Sigma^2/2) + Psm(k)) + B; --09/25/2017    
    implicit none

    ! input
    integer, intent(in)                     :: dim_kt
    real(8), intent(in)                     :: sigma2, A, B
    real(8), dimension(dim_kt), intent(in)  :: Pk_linw, Pk_sm, k_t
    
    ! output
    real(8), dimension(dim_kt), intent(out) :: Pk_model
    
    ! local
    integer :: i
    
    do i=1, dim_kt
        Pk_model(i) = A * ((Pk_linw(i) - Pk_sm(i))*exp(-k_t(i)*k_t(i)*sigma2/2.0) + Pk_sm(i)) + B
    end do
    
    return
end subroutine cal_Pk_model


subroutine lnprior(theta, params_indices, fix_params, lp, dim_theta, dim_params)

    ! lnPrior    
    implicit none
    
    integer, intent(in)                       :: dim_theta, dim_params    
    real(8), dimension(dim_theta), intent(in)  :: theta
    real(8), dimension(dim_params), intent(in) :: params_indices, fix_params
        
    ! output
    real(8), intent(out) :: lp
    
    ! local    
    real(8), dimension(dim_params) :: params_array    
    real(8) :: alpha, sigma2, A, B
    
    ! match params
    call match_params(theta, params_indices, fix_params, params_array, dim_theta, dim_params)
    
    
    alpha  = params_array(1)
    sigma2 = params_array(2)  
    A      = params_array(3)
    B      = params_array(4)

    if (alpha > 0.8d0 .and. alpha<1.2d0 &
        .and. sigma2>-0.1d0 .and. sigma2<200.d0 &
        .and. A > 0.d0 .and. A<1.5d0 &
        .and. B > -1.0d4 .and. B < 1.0d4) then
        lp = 0.d0
    else
        lp = -1.d30  ! return a negative infinitely large number
    endif
    !print*, lp
    return
end subroutine lnprior