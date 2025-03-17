#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements the legendere polynomials, both in
# unscaled (orthogonal) and unscaled (orthonormal) versions.
# the methods are used throughout this project.
#
#


import jax.numpy as jnp
import jax

def legendre(
        n: int,
        tt: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    calculates the values of the legendre
    polynomials at the points specified in tt
    up to order n

    in other words, the function returns an array

    A = [[P_0(tt[0]), ..., P_0(tt[-1])],
          ...
         [P_n(tt[0]), ..., P_n(tt[-1])]]

    where A_kj = P_k(tt_j). we have A.shape = (n+1, tt.shape)

    the method returns an additional array
    containing the derivatives at these points

    B = [[P_0'(tt[0]), ..., P_0'(tt[-1])],
          ...
         [P_n'(tt[0]), ..., P_n'(tt[-1])]]

    where B_kj = P_k'(tt_j). we have B.shape = (n+1, tt.shape)

    the method uses the three term recursions

    (k+1) P_{k+1}(t) = (2k+1) t P_k(t) - k P_{k-1}(t)
    d/dt P_{k+1}(t) = (k+1) P_k(t) + t * d/dt P_k(t)

    with P_0(t) = 1 , P_1(t) = t, P_0'(t) = 0, P_1'(t) = 1

    :param n: maximum degree of legendre polynomials
    :param tt: points to evaluate polynomials at
    :return: arrays A and B containing the values and derivative values
    """

    values_init = jnp.zeros((n+1,) + tt.shape)
    dvalues_init = jnp.zeros((n+1,) + tt.shape) # values of derivative

    P0 = jnp.ones(tt.shape)
    P1 = tt

    # derivatives
    dP0 = jnp.zeros(tt.shape)
    dP1 = jnp.ones(tt.shape)

    values_init = values_init.at[0].set(P0)
    values_init = values_init.at[1].set(P1)
    dvalues_init = dvalues_init.at[0].set(dP0)
    dvalues_init = dvalues_init.at[1].set(dP1)

    ### normal python
    # for k in range(1,n):
    #
    #     # handle normal values
    #     Pkm1, Pk = values[k-1,:], values[k,:]
    #     Pkp1 = (2*k+1)/(k+1) * tt * Pk - k/(k+1) * Pkm1
    #     values = values.at[k+1,:].set(Pkp1)
    #
    #     # handle derivative values
    #     dPk = dvalues[k,:]
    #     dPkp1 = (k+1)*Pk + tt * dPk
    #     dvalues = dvalues.at[k+1,:].set(dPkp1)

    ### for jax.lax
    def loop(k,tup):
        values, dvalues = tup

        # handle normal values
        Pkm1, Pk = values[k-1], values[k]
        Pkp1 = (2*k+1)/(k+1) * tt * Pk - k/(k+1) * Pkm1
        values = values.at[k+1].set(Pkp1)

        # handle derivative values
        dPk = dvalues[k]
        dPkp1 = (k+1)*Pk + tt * dPk
        dvalues = dvalues.at[k+1].set(dPkp1)

        return values, dvalues

    return jax.lax.fori_loop(
        lower=1,
        upper=n,
        body_fun=loop,
        init_val=(values_init, dvalues_init),
        )

def scaled_legendre(
        n: int,
        tt: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    similar to the method `legendre`, but returns
    the values of scaled legendre polynomials

    p_k = sqrt((2*k + 1) / 2) * P_k

    :param n: maximum degree of legendre polynomials
    :param tt: points to evaluate polynomials at
    :return: arrays A and B containing the values and derivative values
    """

    values, dvalues = legendre(n,tt)

    scalings = jnp.sqrt((2*jnp.arange(n+1)+1)/2)

    scaled_values = jnp.einsum('N,N...->N...',scalings,values)
    scaled_dvalues = jnp.einsum('N,N...->N...',scalings,dvalues)

    ### old code, equivalent but slower
    # for j in range(n+1):
    #     values = values.at[j,:].set(scalings[j]*values[j,:])
    #     dvalues = dvalues.at[j,:].set(scalings[j]*dvalues[j,:])

    return scaled_values, scaled_dvalues

def shift_to_interval(
        f_values_df_values: tuple[jnp.ndarray, jnp.ndarray],
        interval: tuple | jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    this function scales the function values provided in
    fvalues to the corresponding values on [a,b] in such
    a way that if fvalues comes from a function f that is
    orthonormal on [-1,1] w.r.t. the L2 inner product,
    then the computed function values belong to the
    function that is scaled and shifted to be orthonormal
    w.r.t. the L2 inner product on [a,b].

    the transformed function reads as

    sqrt(2/(b-a)) f( (2t-a-b) / (b-a) )

    and the derivative of the transformed function reads as

    sqrt(2/(b-a)) * 2/(b-a) * f'( (2t-a-b) / (b-a) )

    the inner transformation in f is not relevant here.
    only the outer scaling is important.

    :param f_values_df_values: values of f and its derivative df (f orthonormal on [-1,1])
    :param interval: interval to transform to
    :return: transformed fvalues according to the formula
    """

    f_values, df_values = f_values_df_values
    a = interval[0]
    b = interval[1]
    scaling = jnp.sqrt(2/(b-a))
    derivative_scaling = scaling * 2/(b-a) # additional chain rule factor for derivatives

    return scaling * f_values, derivative_scaling * df_values

def scaled_shifted_legendre(
        n: int,
        tt: jnp.ndarray,
        interval: tuple | jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    this function calculates the values of the scaled
    legendre polynomials on the interval [a,b] at the
    points specified in tt.
    the polynomials are scaled so that they form an
    orthonormal basis on the interval [a,b].

    :param n: maximum degree of legendre polynomials
    :param tt: points to evaluate polynomials at
    :param interval: interval to shift to
    :return: tuple of array of values of scaled legendre polynomials and derivatives, shape (n+1, tt.shape)
    """

    a, b = interval[0], interval[1]

    # shift tt values to [-1,1]
    tt_shift = (2*tt - a - b)/(b - a)

    # compute scaled legendre values on [-1, 1]
    scaled_legendre_tt_shift = scaled_legendre(n,tt_shift)

    return shift_to_interval(scaled_legendre_tt_shift, interval)

if __name__ == '__main__':

    # jax x64 flag
    jax.config.update("jax_enable_x64", True)

    # plot
    import numpy as np
    import matplotlib.pyplot as plt
    from helpers.other import mpl_settings
    mpl_settings()

    t0 = -2.
    t1 = 2.
    tested_interval = (t0,t1)
    tested_points = jnp.linspace(t0,t1,1000)
    degree = 5
    pk_values, dpk_values = scaled_shifted_legendre(degree,tested_points, tested_interval)

    # plot
    for degree in range(degree+1):
        plt.plot(tested_points, pk_values[degree,:], label=f'$p_{degree}$')

    plt.legend()
    plt.title('scaled and shifted legendre polynomials')
    plt.tight_layout()
    plt.show()

    # plot derivatives
    for degree in range(degree+1):
        plt.plot(tested_points, dpk_values[degree,:], label=f'$p_{degree}^\\prime$')

    plt.legend()
    plt.title('derivative of scaled and shifted legendre polynomials')
    plt.tight_layout()
    plt.show()

    # sanity check for derivative
    print(f'\n\nsanity check for derivative\n')
    check_degree = 2
    check_point_index = 100
    xm1 = tested_points[check_point_index-1]
    x1 = tested_points[check_point_index+1]
    pkm1 = pk_values[check_degree,check_point_index-1]
    pk1 = pk_values[check_degree,check_point_index+1]
    dpk0 = dpk_values[check_degree,check_point_index]
    dpk0_approx = (pk1-pkm1)/(x1-xm1)
    print(f'{dpk0 = }')
    print(f'{dpk0_approx = }')
    print(f'{dpk0 - dpk0_approx = }')


    # test orthonormality
    print(f'\n\ntesting orthonormality\n')
    from scipy.special import roots_legendre
    from gauss import gauss_quadrature_with_values

    num_gauss_points = 10
    gauss_points, gauss_weights = roots_legendre(num_gauss_points)
    gauss_points = jnp.array(gauss_points)
    gauss_weights = jnp.array(gauss_weights)
    # shifted_gauss_points = (2*gauss_points - tested_interval[0] - tested_interval[1])/(tested_interval[1] - tested_interval[0])
    shifted_gauss_points = (t1-t0)/2 * gauss_points + (t0+t1)/2 # shift from [-1, 1] to [t0,t1]
    pk_values, dpk_values = scaled_shifted_legendre(degree,shifted_gauss_points, tested_interval)

    pk_pk_values = jnp.einsum('mt,mt->mt', pk_values, pk_values)
    L2_norms_pk = gauss_quadrature_with_values(gauss_weights, pk_pk_values, interval=tested_interval)
    if jnp.allclose(L2_norms_pk, jnp.ones((degree+1,))):
        print(f'orthonormal!')
    else:
        print(f'WARNING not orthonormal WARNING')
    print(f'{L2_norms_pk = }')




