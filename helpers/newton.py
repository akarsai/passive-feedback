#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements a fast newton method using jax.
#


# jax
import jax.numpy as jnp
from jax import jit, jacobian
import jax.lax

# helpers
from helpers.other import style

@jax.profiler.annotate_function
def newton(
        f: callable,
        Df: callable = None,
        max_iter: int = 10,
        tol: float = 1e-14,
        use_stopping_criterion: bool = False,
        debug: bool = False,
        debug_info_str: str = 'newton',
        ):
    """
    calculates the derivative of f with jax.jacobian
    and returns a jitted newton solver for solving

        f(x) = 0.

    this newton solver can then be called with arbitrary
    initial guesses (and other arguments passed to f)

    the method assumes that the argument of f for which
    we want to find the root is the first one. in other
    words, if

        f = f(a,b,c)

    then for fixed b and c, the method finds a such that

        f(a,b,c) = 0.

    the newton iteration stops after 10 steps and does not
    check convergence.
    (however, a code snippet to check convergence is supplied.)

    :param f: function to find root of (w.r.t. first argument)
    :param Df: jacobian of f (w.r.t. first argument)
    :param max_iter: maximum number of iterations for newton solver
    :param tol: tolerance for newton solver
    :param debug: debug flag
    :return: callable newton solver
    """

    if Df is None:
        Df =  jacobian( f, argnums = 0 )

    @jax.profiler.annotate_function
    @jit
    def solver_with_stopping_criterion(x0, *args, **kwargs):
        """
        this function is a newton solver and can be used
        to find x such that

        f(x, *args, **kwargs) = 0.

        :param x0: initial guess for x0
        :param args: arguments passed to f
        :param kwargs: keyword arguments passed to f
        :return: approximate solution x
        """

        # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):

        # implementation with stopping criteria
        @jax.profiler.annotate_function
        def body(tup):
            i, x = tup
            update = jnp.linalg.solve(Df(x, *args, **kwargs), f(x, *args, **kwargs))
            return  i+1, x - update

        def cond( tup ):
            i, x = tup

            # return jnp.less( i, maxIter )  # only check for maxIter

            return  jnp.logical_and( # check maxIter and tol
                jnp.less(i, max_iter),  # i < maxIter
                jnp.greater(jnp.linalg.norm(f(x, *args, **kwargs)), tol)  # norm( f(x) ) > tol
            )

        i, x = jax.lax.while_loop(cond, body, (0, x0))

        if debug:
            jax.debug.print(f'{style.info}[{debug_info_str}]    ||f(x)|| = {{norm}}    {{iter}} iterations {style.end}', norm = jnp.linalg.norm(f(x, *args, **kwargs)), iter = i)
            # jax.debug.print(f'{style.info}[newton] cond(Df(x)) = {{cond}}{style.end}', cond = jnp.linalg.cond(Df(x, *args, **kwargs)))
            # jax.debug.print(f'{style.info}[newton] ||Df(x)|| = {{norm}}{style.end}', norm = jnp.linalg.norm(Df(x, *args, **kwargs)))
            # jax.debug.print( f'{style.success}[newton] ||f(x)|| = {{norm}} x = jnp.linalg.norm(f(x, * args, ** kwargs )), y = jnp.linalg.cond(Df(x, * args, ** kwargs )))

        return x

    @jax.profiler.annotate_function
    @jit
    def solver_without_stopping_criterion( x0, * args, ** kwargs  ):
        """
        this function is a newton solver and can be used
        to find x such that

        f(x, *args, **kwargs) = 0.

        :param x0: initial guess for x0
        :param args: arguments passed to f
        :param kwargs: keyword arguments passed to f
        :return: approximate solution x
        """

        # implementation without stopping criteria
        def body(i, x):
            ff = f(x, * args, ** kwargs )
            Dff = Df(x, * args, ** kwargs )

            def true_fun():
                if debug: jax.debug.print(f'{style.warning}[{debug_info_str}] linear system is singular, no update{style.end}')
                return jnp.zeros_like(x)

            def false_fun():
                return jnp.linalg.solve(Dff, ff)

            update = jax.lax.cond(jnp.isnan(jnp.linalg.norm(Dff)), true_fun, false_fun)

            # if debug:
            #     jax.debug.print(f'{style.warning}[newton]    ||f(x)|| = {{norm}}{style.end}', norm = jnp.linalg.norm(ff))
            #     jax.debug.print(f'{style.warning}[newton]   ||Df(x)|| = {{norm}}{style.end}', norm = jnp.linalg.norm(Dff))
            #     jax.debug.print(f'{style.warning}[newton] cond(Df(x)) = {{cond}}{style.end}', cond = jnp.linalg.cond(Dff))

            return x - update

        x = jax.lax.fori_loop(0, max_iter, body, x0)

        if debug:
            jax.debug.print(f'{style.info}[{debug_info_str}]    ||f(x)|| = {{norm}}{style.end}', norm = jnp.linalg.norm(f(x, *args, **kwargs)))
            # jax.debug.print(f'{style.info}[newton] cond(Df(x)) = {{cond}}{style.end}', cond = jnp.linalg.cond(Df(x, *args, **kwargs)))
            # jax.debug.print(f'{style.info}[newton] ||Df(x)|| = {{norm}}{style.end}', norm = jnp.linalg.norm(Df(x, *args, **kwargs)))
            # jax.debug.print( f'{style.success}[newton] ||f(x)|| = {{norm}} x = jnp.linalg.norm(f(x, * args, ** kwargs )), y = jnp.linalg.cond(Df(x, * args, ** kwargs )))

        return x

    if use_stopping_criterion:
        solver = solver_with_stopping_criterion
    else:
        solver = solver_without_stopping_criterion

    return solver


