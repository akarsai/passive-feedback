#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements gauss-quadratures used throughout this
# project
#
#


import jax
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from scipy.special import roots_legendre

def gauss_points_and_weights(
        num_gauss_points: int
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    calculates the gauss points and weights for the gauss quadrature
    on [-1,1] with n points using scipy functions

    :param n: number of points
    :return: gauss points and weights
    """

    gauss_points, gauss_weights = roots_legendre(num_gauss_points)
    gauss_points = jnp.array(gauss_points)
    gauss_weights = jnp.array(gauss_weights)


    return gauss_points, gauss_weights


@jax.jit
def gauss_quadrature_with_values(
        gauss_weights: jnp.ndarray,
        fvalues: jnp.ndarray,
        axis: int = -1,
        interval: tuple | jnp.ndarray | None = None,
        length: float | None = None,
        ) -> float | jnp.ndarray:
    """
    uses gauss quadrature to calculate the integral of
    f: R -> R^D (or R^{D,D} or something else)
    on the interval [t0,t1], where

    t0, t1 = interval

    the formula is

    int_t0^t1 f(x) dx ~=~ (t1-t0)/2 sum_i=1^n w_i f( (t1-t0)/2 x_i + (t0+t1)/2 )

    where w_i and x_i are the weights and points for the gauss quadrature
    on [-1,1]

    :param fvalues: values of function on the transformed gauss points (t1-t0)/2 x_i + (t0+t1)/2
    :param interval: interval to calculate integral on (optional)
    :param length: length of the interval (optional, needed if interval is not provided)
    :return: projection calculated with gauss quadrature
    """

    if interval is not None: # length is overwritten
        t0, t1 = interval[0], interval[1]
        length = (t1-t0)
    # else length has to be provided

    # if axis == 1:
    #     integral = length/2 * jnp.einsum('a,a...->...', gauss_weights, fvalues)
    # elif axis == -1:
    #     integral = length/2 * jnp.einsum('a,...a->...', gauss_weights, fvalues)
    # else:
    #     raise ValueError('axis must be 0 or -1 (first or last)')

    # warning: assumes that the integration axis is the last one
    integral = length/2 * jnp.einsum('a,...a->...', gauss_weights, fvalues)

    return integral


if __name__ == '__main__':

    jax.config.update("jax_debug_nans", True)

    # see legendre.py for example usage

    pass