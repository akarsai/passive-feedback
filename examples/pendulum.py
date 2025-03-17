#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements a nonlinear pendulum as a port-hamiltonian system
#
#

# jax
import jax.numpy as jnp
from jax import grad

# nonlinear system class
from helpers.system import NonlinearAffineSystem

class Pendulum(NonlinearAffineSystem):

    def __init__(
            self,
            initial_state = None,
            friction_coefficient = 0.2,
            gravitation = 9.81,
            ):
        """
        models a nonlinear pendulum with friction and velocity control

        the energies of the system is E = E_kin + E_pot with

        E_kin = 1/2 * m * l^2 * theta_dot^2
        E_pot = m * g * l * (1 - cos(theta)).

        we take m = l = 1 in the following.

        the dynamics read as

        d^2/dt^2 theta = - g * sin(theta) - friction_coefficient * d/dt theta + u

        after order reduction, we arrive at

        d/dt theta = omega
        d/dt omega = - g * sin(theta) - friction_coefficient * omega + u

        setting z = (theta, omega), we arrive at a port-hamiltonian representation

        d/dt z = (J - R) eta(z) + B u

        Args:
            initial_state: (optional) initial state of the pendulum
            friction_coefficient: (optional) friction coefficient
        """

        self.gravitation = gravitation

        self.J = jnp.array([[0, 1], [-1, 0]])
        self.R = jnp.array([[0, 0], [0, friction_coefficient]])
        self.B = jnp.array([[0],[1]])

        self.hamiltonian = lambda z: self.gravitation * (1-jnp.cos(z[0])) + 1/2 * z[1]**2
        self.eta = lambda z: jnp.array([self.gravitation * jnp.sin(z[0]), z[1]])
        self.r = lambda v: - (self.J - self.R) @ v # map f(z) = -r(eta(z))

        if initial_state is None:
            initial_state = jnp.array([1/4 * jnp.pi, -1.0])

        super().__init__(
            f = lambda z: (self.J - self.R) @ self.eta(z),
            g = lambda z: self.B,
            h = lambda z: self.B.T @ self.eta(z),
            # h = lambda z: jnp.array([[1], [0]]).T @ z, # for other measurement y
            initial_state = initial_state,
            ncontrol = 1
            )

        return

if __name__ == '__main__':

    # jax config
    import jax
    jax.config.update("jax_enable_x64", True)

    # time integration
    from helpers.time_integration import bdf4

    # setup pendulum
    pendulum = Pendulum()

    # setup control
    control = jax.vmap(lambda t: jnp.array([0.5]), in_axes=0)

    # simulate pendulum
    final_time = 10
    nt = 1000
    tt = jnp.linspace(0, final_time, nt)
    zz = bdf4(pendulum.dynamics, tt, pendulum.initial_state, control(tt))

    # plot
    import matplotlib.pyplot as plt
    from helpers.other import mpl_settings
    mpl_settings()

    plt.plot(tt, zz[:,0], label=r'$\theta$')
    plt.plot(tt, zz[:,1], label=r'$\dot\theta$')
    plt.xlabel('time')
    plt.legend()
    plt.title('nonlinear pendulum')
    plt.tight_layout()
    plt.show()

