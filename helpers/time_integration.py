#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements time integration methods suitable for
# automatic differentiation via jax
#
# currently, the following methods are implemented:
# - implicit euler (order of convergence 1)
# - implicit midpoint with linear interpolation of the
#   control (crank nicolson, order of convergence 2)
# - a custom discrete gradient method suitable for
#   passive systems
# - bdf4 (order of convergence 4)
#

# jax
import jax
import jax.numpy as jnp
from jax import jit, jacobian
import jax.lax

# custom imports
from helpers.newton import newton
from helpers.other import style

@jax.profiler.annotate_function
def implicit_euler(
        f: callable,
        tt: jnp.ndarray,
        z0: jnp.ndarray,
        uu: jnp.ndarray,
        type = 'forward',
        debug = False,
        ) -> jnp.ndarray:

    """
    uses implicit euler method to solve the initial value problem

    z' = f(z,u), z(tt[0]) = z0    (if type == 'forward')

    or

    p' = f(p,u), p(tt[-1]) = p0   (if type == 'backward')

    :param f: right hand side of ode, f = f(z,u)
    :param tt: timepoints, assumed to be evenly spaced
    :param z0: initial or final value
    :param uu: control input at timepoints, shape = (len(tt), N)
    :param type: 'forward' or 'backward'
    :param debug: if True, print debug information
    :return: solution of the problem in the form
        z[i,:] = z(tt[i])
        p[i,:] = p(tt[i])
    """

    N = len(z0) # system dimension
    nt = len(tt) # number of timepoints
    dt = tt[1] - tt[0] # timestep, assumed to be constant

    # identity matrix
    eye = jnp.eye(N)

    # jacobian of f
    Df = jacobian(f, argnums=0)

    @jax.profiler.annotate_function
    def F_implicit_euler(zj, zjm1, uj):
        return \
            zj \
            - zjm1 \
            - dt*f(zj,uj)

    @jax.profiler.annotate_function
    def DF_implicit_euler( zj, zj1, uj ):
        return eye - dt*Df(zj,uj)

    solver = newton(F_implicit_euler, Df=DF_implicit_euler, debug=debug)

    if type == 'forward':

        z = jnp.zeros((nt,N))
        z = z.at[0,:].set(z0)

        @jax.profiler.annotate_function
        def body( j, var ):
            z, uu = var

            zjm1 = z[j-1,:]
            uj = uu[j,:]

            y = solver(zjm1, zjm1, uj)
            z = z.at[j,:].set(y)

            if debug:
                jax.debug.print( 'j = {x}', x = j)

            return z , uu

        z,_ = jax.lax.fori_loop(1, nt, body, (z,uu))

        return z

    else: # type == 'backward'

        return implicit_euler(f, tt[::-1], z0, uu[::-1,:], type='forward')[::-1,:]


@jax.profiler.annotate_function
def implicit_midpoint(
        f: callable,
        tt: jnp.ndarray,
        z0: jnp.ndarray,
        uu: jnp.ndarray,
        type = 'forward',
        debug = False,
        ) -> jnp.ndarray:

    """
    uses implicit midpoint method to solve the initial value problem

    z' = f(z,u), z(tt[0]) = z0    (if type == 'forward')

    or

    p' = f(p,u), p(tt[-1]) = p0   (if type == 'backward')

    in the implementation, the control input is linearly interpolated
    to evaluate at midpoints of the time interval.

    :param f: right hand side of ode, f = f(z,u)
    :param tt: timepoints, assumed to be evenly spaced
    :param z0: initial or final value
    :param uu: control input at timepoints, shape = (len(tt), N)
    :param type: 'forward' or 'backward'
    :param debug: if True, print debug information
    :return: solution of the problem in the form
        z[i,:] = z(tt[i])
        p[i,:] = p(tt[i])
    """

    N = len(z0) # system dimension
    nt = len(tt) # number of timepoints
    dt = tt[1] - tt[0] # timestep, assumed to be constant
    uumid = 1/2 * (uu[1:,:] + uu[:-1,:]) # linear interpolation of control input

    @jax.profiler.annotate_function
    def F_implicit_midpoint(zj, zjm1, uj12):
        return \
            zj \
            - zjm1 \
            - dt*f( 1/2*(zjm1+zj), uj12)

    eye = jnp.eye(N)
    Df = jacobian(f, argnums=0)

    @jax.profiler.annotate_function
    def DF_implicit_midpoint(zj, zjm1, uj12):
        return eye - 1/2 * dt * Df( 1/2 * (zjm1 + zj) , uj12)

    solver = newton(F_implicit_midpoint, debug=debug)

    if type == 'forward':

        z = jnp.zeros((nt,N))
        z = z.at[0,:].set(z0)

        @jax.profiler.annotate_function
        # after that bdf method
        def body( j, var ):
            z, uumid = var

            zjm1 = z[j-1,:]
            uj12 = uumid[j-1,:]

            y = solver(zjm1, zjm1, uj12)
            z = z.at[j,:].set(y)

            if debug: jax.debug.print(f'{style.info}timestep number = {{j}}{style.end}', j=j)
            # jax.debug.print( 'iter = {x}', x = i)

            # jax.debug.print('\n forward bdf: j = {x}', x = j)

            # jax.debug.print('log10(||residual||) = {x}', x = jnp.log10(jnp.linalg.norm(m_bdf(y,zjm1,zjm2,zjm3,zjm4,uj))) )

            return z, uumid

        z, _ = jax.lax.fori_loop(1, nt, body, (z,uumid))

        return z

    else: # type == 'backward'

        return implicit_midpoint(f, tt[::-1], z0, uu[::-1,:], type='forward')[::-1,:]

def discrete_gradient(
        r: callable,
        ham_eta: callable,
        B: callable,
        tt: jnp.ndarray,
        z0: jnp.ndarray,
        uu: jnp.ndarray,
        debug: bool = False,
        return_hamiltonian: bool = False,
    ) -> jnp.array:
    """
    computes an approximate solution of

    z' = -r(eta(z)) + B(z) u,    z(0) = z_0

    on given timesteps in the time horizon [0,T].
    the system describes passive dynamics, where
    the function r satisfies

    eta(z)^T r(eta(z)) >= 0 for all z

    and eta is the gradient of the hamiltonian function.

    the method uses the discrete gradient pair from
    the paper https://arxiv.org/abs/2311.00403

    :param r: function r in the system dynamics
    :param ham_eta: function returning (hamiltonian(z), eta(z)) for given z
    :param B: matrix B in the system dynamics
    :param tt: array of timepoints to be used
    :param z0: initial condition
    :param uu: value of control input at timepoints
    :param debug: (optional) debug flag
    :param return_hamiltonian: (optional) if True, also the hamiltonian values are returned
    :return: values of solution at time points [t_0, t_1, ... ]
    """

    nt = tt.shape[0]
    nsys = z0.shape[0]
    Delta_t = tt[1] - tt[0] # assumed to be constant

    uumid = 1/2 * (uu[1:,:] + uu[:-1,:]) # linear interpolation of control input

    def get_eta_bar(z, zhat, ham_z):
        # computes eta_bar
        # tries to minimize function calls to ham_eta

        # _, ham_z, _ = ham_eta(zhat) # hamiltonian(z)
        ham_zhat, _ = ham_eta(zhat) # hamiltonian(zhat)
        _, eta_mid = ham_eta(1/2 * (z + zhat)) # eta(1/2*(z+zhat))

        alpha1 = ham_zhat - ham_z - eta_mid.T @ (zhat - z)
        alpha2 = (zhat - z).T @ (zhat - z)

        def true_fun():
            return eta_mid

        def false_fun():
            return eta_mid + alpha1/alpha2 * (zhat - z)

        return jax.lax.cond(jnp.allclose(alpha2, 0.0), true_fun, false_fun)

    def F(zkp1, zk, ham_zk, umid):

        return (
            zkp1
            - zk
            + Delta_t * r(get_eta_bar(zk, zkp1, ham_zk))
            - Delta_t * B(1/2 * (zk + zkp1)) @ umid
        )

    solver = newton(F, debug=debug)

    # set initial condition and hamiltonian array
    z = jnp.zeros((nt,nsys)).at[0,:].set( z0 )
    ham_zk, _ = ham_eta(z0)
    ham = jnp.zeros((nt,)).at[0].set(ham_zk)

    # loop
    def body( k, var ):
        z, ham, uumid = var

        zk = z[k,:]
        ham_zk = ham[k]
        umidk = uumid[k,:]

        y = solver(zk, zk, ham_zk, umidk)
        z = z.at[k+1,:].set(y)
        ham_zk, _ = ham_eta(y) # hamiltonian(zkp1)
        ham = ham.at[k+1].set(ham_zk)

        if debug: jax.debug.print('timestep number = {k}', k=k)

        return z, ham, uumid

    z, ham, _ = jax.lax.fori_loop(0, nt-1, body, (z, ham, uumid))

    if return_hamiltonian:
        return z, ham

    return z


@jax.profiler.annotate_function
def bdf4(
        f: callable,
        tt: jnp.ndarray,
        z0: jnp.ndarray,
        uu: jnp.ndarray,
        type = 'forward',
        debug = False,
        ) -> jnp.ndarray:

    """
    uses bdf4 method to solve the initial value problem

    z' = f(z,u), z(tt[0]) = z0    (if type == 'forward')

    or

    p' = f(p,u), p(tt[-1]) = p0   (if type == 'backward')

    :param f: right hand side of ode, f = f(z,u)
    :param tt: timepoints, assumed to be evenly spaced
    :param z0: initial or final value
    :param uu: control input at timepoints, shape = (len(tt), N)
    :param type: 'forward' or 'backward'
    :param debug: if True, print debug information
    :return: solution of the problem in the form
        z[i,:] = z(tt[i])
        p[i,:] = p(tt[i])
    """

    N = len(z0) # system dimension
    nt = len(tt) # number of timepoints
    dt = tt[1] - tt[0] # timestep

    # identity matrix
    eye = jnp.eye(N)

    # jacobian of f
    Df = jacobian(f, argnums = 0)

    @jax.profiler.annotate_function
    def F_bdf(zj, zj1, zj2, zj3, zj4 , uj):
        return\
            25*zj \
            - 48*zj1 \
            + 36*zj2 \
            - 16*zj3 \
            + 3*zj4 \
            - 12*dt*f(zj,uj)

    @jax.profiler.annotate_function
    def DF_bdf(zj, zj1, zj2, zj3, zj4 , uj):
        return 25*eye - 12*dt*Df(zj,uj)

    # for first four values
    @jax.profiler.annotate_function
    def F_start(z1234):
        # the magic coefficients in this function come from a polynomial approach
        # the approach calculates 4 timesteps at once and is of order 4.
        # for details, see here: https://colab.research.google.com/drive/1DHtzD3U1PsMQbn-nsBF3crh_Tj3cgmMd?usp=sharing

        z1 = z1234[:N]
        z2 = z1234[N:2*N]
        z3 = z1234[2*N:3*N]
        z4 = z1234[3*N:]

        # entries of F
        pprime_t1 = -3.0*z0 - 10.0*z1 + 18.0*z2 - 6.0*z3 + z4
        pprime_t2 = z0 - 8.0*z1 + 8.0*z3 - 1.0*z4
        pprime_t3 = -1.0*z0 + 6.0*z1 - 18.0*z2 + 10.0*z3 + 3.0*z4
        pprime_t4 = 3.0*z0 - 16.0*z1 + 36.0*z2 - 48.0*z3 + 25.0*z4

        return jnp.hstack((
                pprime_t1 - 12*dt*f(z1,uu[1,:]),
                pprime_t2 - 12*dt*f(z2,uu[2,:]),
                pprime_t3 - 12*dt*f(z3,uu[3,:]),
                pprime_t4 - 12*dt*f(z4,uu[4,:])
            ))

    @jax.profiler.annotate_function
    def DF_start(z1234):
        # the magic coefficients in this function come from a polynomial approach
        # the approach calculates 4 timesteps at once and is of order 4.
        # for details, see here: https://colab.research.google.com/drive/1DHtzD3U1PsMQbn-nsBF3crh_Tj3cgmMd?usp=sharing

        z1 = z1234[:N]
        z2 = z1234[N:2*N]
        z3 = z1234[2*N:3*N]
        z4 = z1234[3*N:]

        # first row
        DF_11 = -10.0 * eye - 12*dt*Df(z1,uu[1,:])
        DF_12 = 18.0 * eye
        DF_13 = -6.0 * eye
        DF_14 = 1.0 * eye
        DF_1 = jnp.hstack((DF_11,DF_12,DF_13,DF_14))

        # second row
        DF_21 = -8.0 * eye
        DF_22 = 0.0 * eye - 12*dt*Df(z2,uu[2,:])
        DF_23 = 8.0 * eye
        DF_24 = -1.0 * eye
        DF_2 = jnp.hstack((DF_21,DF_22,DF_23,DF_24))

        # third row
        DF_31 = 6.0 * eye
        DF_32 = -18.0 * eye
        DF_33 = 10.0 * eye - 12*dt*Df(z3,uu[3,:])
        DF_34 = 3.0 * eye
        DF_3 = jnp.hstack((DF_31,DF_32,DF_33,DF_34))

        # fourth row
        DF_41 = -16.0 * eye
        DF_42 = 36.0 * eye
        DF_43 = -48.0 * eye
        DF_44 = 25.0 * eye - 12*dt*Df(z4,uu[4,:])
        DF_4 = jnp.hstack((DF_41,DF_42,DF_43,DF_44))

        # return all rows together
        return jnp.vstack((DF_1,DF_2,DF_3,DF_4))

    solver_start = newton(F_start, Df=DF_start, debug=debug)
    solver_bdf = newton(F_bdf, Df=DF_bdf, debug=debug)

    if type == 'forward':

        z = jnp.zeros((nt,N))
        z = z.at[0,:].set(z0)

        # first few steps with polynomial interpolation technique
        z1234 = solver_start(jnp.hstack((z0,z0,z0,z0)))
        z = z.at[1,:].set(z1234[:N])
        z = z.at[2,:].set(z1234[N:2*N])
        z = z.at[3,:].set(z1234[2*N:3*N])

        @jax.profiler.annotate_function
        # after that bdf method
        def body( j, var ):
            z, uu = var

            zjm4 = z[j-4,:]
            zjm3 = z[j-3,:]
            zjm2 = z[j-2,:]
            zjm1 = z[j-1,:]
            uj   = uu[j,:]

            y = solver_bdf(zjm1, zjm1, zjm2, zjm3, zjm4, uj)
            z = z.at[j,:].set(y)


            if debug:
                jax.debug.print('j = {x}', x = j)
            # jax.debug.print( 'iter = {x}', x = i)

            # jax.debug.print('\n forward bdf: j = {x}', x = j)

            # jax.debug.print('log10(||residual||) = {x}', x = jnp.log10(jnp.linalg.norm(m_bdf(y,zjm1,zjm2,zjm3,zjm4,uj))) )

            return z , uu

        z,_ = jax.lax.fori_loop(4, nt, body, (z,uu))

        return z

    else: # type == 'backward'

        return bdf4(f, tt[::-1], z0, uu[::-1,:], type='forward')[::-1,:]



if __name__ == '__main__':


    # below is a simple convergence test to confirm the
    # theoretical order of convergence

    import matplotlib.pyplot as plt
    jax.config.update("jax_enable_x64", True) # double precision

    from other import mpl_settings
    mpl_settings()

    # timer
    from timeit import default_timer as timer

    n  = 5
    T = 10

    A = -jnp.eye(n)
    B = jnp.zeros((n,2))
    B = B.at[0,0].set(1.0)
    B = B.at[-1,-1].set(-1.0)
    hamiltonian = lambda z: 1/2 * z.T @ z # for discrete gradient method
    eta = jax.grad(hamiltonian) # for discrete gradient method
    # r = lambda v: v # for discrete gradient method
    z0 = jnp.ones((n,))

    def r(v):
        return v

    def ham_eta(z):
        return hamiltonian(z), eta(z)

    def f(z,u):
        return A@z + B@u

    def control(t):
        # return jnp.zeros((t.shape[0],2))
        return jnp.array([jnp.sin(t), jnp.cos(t)])
    control = jax.vmap(control, in_axes=0, out_axes=0,) # 0 = index where time is


    # general settings
    base_Delta_t = 1e-3
    num_Delta_t_steps = 9
    Delta_t_array = jnp.array([2**i * base_Delta_t for i in range(num_Delta_t_steps)])
    ref_order_smaller = 3 # by which order of magnitude should the reference solution be smaller than the smallest tested Delta t?
    ref_Delta_t = base_Delta_t/(2**ref_order_smaller) # Delta t for reference solution

    # convert Delta_t values to nt values
    groß_n = (T/Delta_t_array[-1]).astype(int)
    nt_array = jnp.flip(jnp.array([groß_n * 2**(order) + 1 for order in range(num_Delta_t_steps)]))
    nt_ref = 2**(ref_order_smaller+num_Delta_t_steps-1) * groß_n + 1

    # convert back to get "real" Delta_t corresponding to the nt_spp values
    Delta_t_array = T/nt_array
    Delta_t_ref = T/nt_ref

    print(f'\nnt_ref = {nt_ref}\nDelta_t_ref = {Delta_t_ref:e}')
    print(f'\nnt_array = {nt_array}\nDelta_t_array = {Delta_t_array}\n')


    # for method in ['implicit euler', 'implicit midpoint', 'bdf4', 'discrete gradient']:
    for method in ['discrete gradient']:

        if method == 'implicit euler':
            integrator = implicit_euler
        elif method == 'implicit midpoint':
            integrator = implicit_midpoint
        elif method == 'discrete gradient':
            integrator = discrete_gradient
        else:
            integrator = bdf4

        for type in ['forward', 'backward']:

            print(f"\n{type} simulation for z' = Az + Bu with {method} method\n")

            # obtain reference solution with bdf4
            tt_ref = jnp.linspace(0,T,nt_ref)
            uu_ref = control(tt_ref)
            if method == 'discrete gradient':
                zz_ref = discrete_gradient(r, ham_eta, lambda z: B, tt_ref, z0, uu_ref)
            else:
                zz_ref = integrator(f, tt_ref, z0, uu_ref, type=type)


            print(f'reference solution obtained!')

            max_errors = []

            for k in range(num_Delta_t_steps):
                nt = int(nt_array[k])

                tt = jnp.linspace(0,T,nt)
                uu = control(tt)

                if method == 'discrete gradient':
                    zz = discrete_gradient(r, ham_eta, lambda z: B, tt, z0, uu)
                else:
                    zz = integrator(f, tt, z0, uu, type=type)

                # eval reference solution on coarse time gitter
                zz_ref_resampled = zz_ref[::2**(k+ref_order_smaller),:]

                diff = zz_ref_resampled - zz

                # calculate relative error
                error = jnp.linalg.norm(diff, axis=1)#/jnp.linalg.norm(zz_bdf, axis=1) # norms along axis 1, since axis 0 are the time points
                error = error/jnp.max(jnp.linalg.norm(zz_ref, axis=1))

                max_error = jnp.max(error) # Linf error
                max_errors.append(max_error)

                print(f'done for nt={nt}')

            plt.loglog(Delta_t_array, max_errors, label=method, marker='o')
            # plt.xticks(nt_array)

            # comparison
            if method == 'implicit euler':
                c1 = max_errors[-2]/Delta_t_array[-2]**1
                firstorder = c1*Delta_t_array**1
                plt.loglog(Delta_t_array, firstorder, label='$(\Delta t)^1$', linestyle='--')
            elif method == 'implicit midpoint' or method == 'discrete gradient':
                c2 = max_errors[-2]/Delta_t_array[-2]**2
                secondorder = c2*Delta_t_array**2
                plt.loglog(Delta_t_array, secondorder, label='$(\Delta t)^2$', linestyle='--')
            else:
                c4 = max_errors[-2]/Delta_t_array[-2]**4
                fourthorder = c4*Delta_t_array**4
                plt.loglog(Delta_t_array, fourthorder, label='$(\Delta t)^4$', linestyle='--')

            plt.xlabel('$\Delta t$')
            plt.ylabel('relative error compared to finest grid')
            plt.title(f'{type} simulation for $\\dot{{z}} = Az + Bu$ with {method} method')
            plt.legend()
            plt.show()

            plt.plot(tt_ref, jnp.linalg.norm(zz_ref, axis=1), label='$\|z(t)\|_2$')
            plt.xlabel('$t$')
            plt.ylabel('$\|z(t)\|_2$')
            plt.title(f'{type} simulation for $\\dot{{z}} = Az + Bu$ with {method} method')
            plt.legend()
            plt.show()


    pass