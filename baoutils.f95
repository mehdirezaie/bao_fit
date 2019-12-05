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



