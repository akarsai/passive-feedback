#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements some visualization helper functions
#
#

# jax
import jax
import jax.numpy as jnp
from jax import jit, jacobian
import jax.lax

# matplotlib
import matplotlib.pyplot as plt

# custom imports
from helpers.other import mpl_settings

def discrete_gradient_powerbalance(
        r: callable,
        ham_eta: callable,
        B: callable,
        tt: jnp.ndarray,
        zz_ham: jnp.ndarray,
        uu: jnp.ndarray,
        relative: bool = True,
        title: str = None,
    ) -> tuple[jnp.array, jnp.ndarray]:
    """
    visualizes the energy balance of the discrete gradient method

    :param r: function r in the system dynamics zdot = -r(eta(z)) + B(z)u
    :param ham_eta: function returning (hamiltonian(z), eta(z)) for given z
    :param B: matrix B in the system dynamics
    :param tt: array of timepoints to be used
    :param zz_ham: solution computed by the discrete gradient method with return_hamiltonian=True
    :param uu: value of control input at timepoints
    :param relative: if True, the relative error is computed
    :return: None
    """

    nt = tt.shape[0]
    Delta_t = tt[1] - tt[0] # assumed to be constant
    uumid = 1/2 * (uu[1:,:] + uu[:-1,:]) # linear interpolation of control input

    def get_eta_bar(z, zhat, ham_z, ham_zhat):
        # computes eta_bar
        # tries to minimize function calls to ham_eta

        # _, ham_z, _ = ham_eta(zhat) # hamiltonian(z)
        # _, ham_zhat, _ = ham_eta(zhat) # hamiltonian(zhat)
        _, eta_mid = ham_eta(1/2 * (z + zhat)) # eta(1/2*(z+zhat))

        alpha1 = ham_zhat - ham_z - eta_mid.T @ (zhat - z)
        alpha2 = (zhat - z).T @ (zhat - z)

        def true_fun():
            return eta_mid

        def false_fun():
            return eta_mid + alpha1/alpha2 * (zhat - z)

        return jax.lax.cond(jnp.allclose(alpha2, 0.0), true_fun, false_fun)

    # energy balance reads as
    #
    # (ham(z_{k+1}) - ham(z_k))/(Delta t)
    #   = eta_bar(z_k, z_{k+1})^T f_bar(z_k, z_{k+1})
    #     + eta_bar(z_k, z_{k+1})^T B(1/2*(z_k+z_{k+1})) u_{k+1/2}

    zz, ham = zz_ham
    zk = zz[0,:]
    errors = jnp.zeros((nt-1,))
    lhss = jnp.zeros((nt-1,))

    # jax.lax.fori implementation
    def body_fun(k, tup):

        zk, ham, errors, lhss = tup

        zkp1 = zz[k+1,:]
        ham_zk = ham[k]
        ham_zkp1 = ham[k+1]
        eta_bar = get_eta_bar(zk, zkp1, ham_zk, ham_zkp1)

        lhs = (ham_zkp1 - ham_zk)/Delta_t
        rhs = - eta_bar.T @ r(eta_bar) + eta_bar.T @ B(1/2 * (zk + zkp1)) @ uumid[k]

        # jax.debug.print('sign of eta_bar.T @ f_bar: {sign}', sign=jnp.sign(eta_bar.T @ f_bar))

        error = jnp.abs(lhs-rhs)

        errors = errors.at[k].set(error)
        lhss = lhss.at[k].set(lhs)

        zk = zkp1

        return zk, ham, errors, lhss

    _, ham, errors, lhss = jax.lax.fori_loop(0, nt-1, body_fun, (zk, ham, errors, lhss))

    if relative:
        errors = errors / jnp.max(jnp.abs(lhss))

    if title is not None: visualize_errors([(tt, errors, title)], title=title)

    return ham, errors

def visualize_hamiltonian(
        tuplist: list[tuple[jnp.ndarray, jnp.ndarray, str]],
        axis_type: str = 'linear',
        title: str = None,
        ylabeltext = r'$\mathcal{H}(z)$',
        savepath: str = None,
        ) -> None:

    if axis_type == 'linear':
        plot_function = plt.plot
    elif axis_type == 'semilogy':
        plot_function = plt.semilogy
    else:
        raise NotImplementedError(f'axis type {axis_type} not implemented')

    for tt, ham, label in tuplist:
        plot_function(tt, ham, label=label)

    plt.legend()
    plt.xlabel('time $t$')
    plt.ylabel(ylabeltext)

    if savepath is not None:
        plt.tight_layout()
        plt.savefig(savepath + '.pgf') # save as pgf
        plt.savefig(savepath + '.png') # save as png
        print(f'figure saved under savepath {savepath} (as pgf and png)')

    plt.title(title)
    plt.tight_layout()
    plt.show()

def visualize_errors(
        tuplist: list[tuple[jnp.ndarray, jnp.ndarray, str]],
        title: str = None,
        ylabeltext = 'relative error in power balance',
        savepath: str = None,
        ) -> None:

    for tt, errors, label in tuplist:
        plt.semilogy(tt[:-1], errors, label=label)

    plt.legend()
    plt.xlabel('time $t$')
    plt.ylabel(ylabeltext)
    plt.ylim([.5e-17, .5e-4])

    if savepath is not None:
        plt.tight_layout()
        plt.savefig(savepath + '.pgf') # save as pgf
        plt.savefig(savepath + '.png') # save as png
        print(f'figure saved under savepath {savepath} (as pgf and png)')

    plt.title(title)
    plt.tight_layout()
    plt.show()


# controller
def plot_controller_trajectory(
        controller_trajectory: jnp.ndarray,
        tt: jnp.ndarray,
        axtitle: str = 'controller',
        variable_name: str = r'\hat{z}',
        title: str = None,
        savepath: str = None,
        ):

    # mpl_settings(figsize=(5.5,2))

    zzhat = controller_trajectory
    nsys = zzhat.shape[1]

    fig, ax = plt.subplots()

    # if nsys > 2:
    #     zzhat_plot = jnp.linalg.norm(zzhat, axis=1)**2
    #     zzhat_legend = f'$\\lVert {variable_name}(t) \\rVert^2$'
    # else:
    zzhat_plot = zzhat
    zzhat_legend = [f'${variable_name}_{{{i+1}}}(t)$' for i in range(nsys)]

    ax.plot(tt, zzhat_plot, label=zzhat_legend)
    ax.set_title(axtitle)
    ax.set_xlabel('time $t$')
    ax.legend()

    if savepath is not None:
        fig.tight_layout()
        plt.savefig(savepath + '.pgf') # save as pgf
        plt.savefig(savepath + '.png') # save as png
        print(f'figure saved under savepath {savepath} (as pgf and png)')

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()

    return


# controller
def plot_state_comparison(
        timepoints_state_labelname_tuple_list: list[tuple[jnp.ndarray, jnp.ndarray, str, any]],
        title: str = 'comparison of state trajectories',
        ylabel: str = r'$\lVert z(t) \rVert^2$',
        semilogy: bool = True,
        savepath: str = None,
        ):

    fig, ax = plt.subplots()

    if semilogy:
        plot = ax.semilogy
    else:
        plot = ax.plot

    for tt, zz, label, color in timepoints_state_labelname_tuple_list:
        plot(tt, jnp.linalg.norm(zz, axis=1)**2, label=label, color=color)

    ax.set_xlabel('time $t$')
    ax.set_ylabel(ylabel)
    ax.legend()

    if savepath is not None:
        fig.tight_layout()
        plt.savefig(savepath + '.pgf') # save as pgf
        plt.savefig(savepath + '.png') # save as png
        print(f'figure saved under savepath {savepath} (as pgf and png)')

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()

    return

def plot_controller_difference(
        controller_trajectory_1: jnp.ndarray,
        controller_trajectory_2: jnp.ndarray,
        tt: jnp.ndarray,
        index_1: str = '1',
        index_2: str = '2',
        relative: bool = True,
        legend_loc = 'best',
        title: str = '',
        savepath: str = None,
        ):

    zz_diff = controller_trajectory_1 - controller_trajectory_2

    fig, ax = plt.subplots(1, 1)
    # ax = ax[0]

    zz_diff_plot = jnp.linalg.norm(zz_diff, axis=1)
    zz_diff_legend = f'$\\lVert \\hat{{z}}_{{\\text{{{index_1}}}}}(t) - \\hat{{z}}_{{\\text{{{index_2}}}}}(t) \\rVert$'

    ax.semilogy(tt, zz_diff_plot, label=zz_diff_legend)
    # ax.set_title('plant')
    ax.set_xlabel('time $t$')
    ax.legend(loc=legend_loc)

    if savepath is not None:
        fig.tight_layout()
        plt.savefig(savepath + '.pgf') # save as pgf
        plt.savefig(savepath + '.png') # save as png
        print(f'figure saved under savepath {savepath} (as pgf and png)')

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()

    return

def plot_coupled_trajectory(
        coupled_trajectory: jnp.ndarray,
        tt: jnp.ndarray,
        plant_trajectory_name: str = r'z',
        controller_trajectory_name: str = r'\hat{z}',
        plant_ylim: tuple = None,
        controller_ylim: tuple = None,
        title: str = '',
        savepath: str = None,
        ):

    nsys = coupled_trajectory.shape[1] // 2

    zz = coupled_trajectory[:, :nsys]
    zzhat = coupled_trajectory[:, nsys:]

    fig, ax = plt.subplots(2, 1)

    # if nsys > 2:
    #     zz_plot = jnp.linalg.norm(zz, axis=1)**2
    #     zz_legend = f'$\\lVert {plant_trajectory_name}(t) \\rVert^2$'
    #     zzhat_plot = jnp.linalg.norm(zzhat, axis=1)**2
    #     zzhat_legend = f'$\\lVert {controller_trajectory_name}(t) \\rVert^2$'
    # else:
    zz_plot, zzhat_plot = zz, zzhat
    zz_legend = [f'${plant_trajectory_name}_{{{i}}}(t)$' for i in range(1,nsys+1)]
    zzhat_legend = [f'${controller_trajectory_name}_{{{i}}}(t)$' for i in range(1,nsys+1)]

    ax[0].plot(tt, zz_plot, label=zz_legend)
    ax[0].set_title('plant')
    # ax[0].set_xlabel('time $t$')
    # ax[0].set_ylabel(r'$\lVert z \rVert^2$')
    if plant_ylim is not None:
        ax[0].set_ylim(plant_ylim)
    ax[0].legend()
    ax[1].plot(tt, zzhat_plot, label=zzhat_legend)
    ax[1].set_title('controller')
    ax[1].set_xlabel('time $t$')
    if controller_ylim is not None:
        ax[1].set_ylim(controller_ylim)
    # ax[1].set_ylabel(r'$\lVert \hat{z} \rVert^2$')
    ax[1].legend()

    if savepath is not None:
        fig.tight_layout()
        plt.savefig(savepath + '.pgf') # save as pgf
        plt.savefig(savepath + '.png') # save as png
        print(f'figure saved under savepath {savepath} (as pgf and png)')

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()

    return

def plot_coupled_difference(
        coupled_trajectory_1: jnp.ndarray,
        coupled_trajectory_2: jnp.ndarray,
        tt: jnp.ndarray,
        index_1: str = '1',
        index_2: str = '2',
        relative: bool = True,
        legend_loc = 'best',
        title: str = '',
        savepath: str = None,
        ):

    nsys = coupled_trajectory_1.shape[1] // 2

    diff_trajectory = coupled_trajectory_1 - coupled_trajectory_2

    zz_diff = diff_trajectory[:, :nsys]
    zzhat_diff = diff_trajectory[:, nsys:]

    fig, ax = plt.subplots(2, 1)

    # if nsys > 2:
    if 1:
        zz_diff_plot = jnp.linalg.norm(zz_diff, axis=1)
        zz_diff_legend = f'$\\lVert z_{{\\text{{{index_1}}}}}(t) - z_{{\\text{{{index_2}}}}}(t) \\rVert$'
        zzhat_diff_plot = jnp.linalg.norm(zzhat_diff, axis=1)
        zzhat_diff_legend = f'$\\lVert \hat{{z}}_{{\\text{{{index_1}}}}}(t) - \hat{{z}}_{{\\text{{{index_2}}}}}(t) \\rVert$'
    # else:
    #     zz_diff_plot, zzhat_diff_plot = jnp.abs(zz_diff), jnp.abs(zzhat_diff)
    #     zz_diff_legend = [f'$| z_{{{i}}}^{{\\text{{{index_1}}}}}(t) - z_{{{i}}}^{{\\text{{{index_2}}}}}(t) | $' for i in range(1,nsys+1)]
    #     zzhat_diff_legend = [f'$| \hat{{z}}_{{{i}}}^{{\\text{{{index_1}}}}}(t) - \hat{{z}}_{{{i}}}^{{\\text{{{index_2}}}}}(t) |$' for i in range(1,nsys+1)]

    ax[0].semilogy(tt, zz_diff_plot, label=zz_diff_legend)
    ax[0].set_title('plant')
    ax[0].set_xlabel('time $t$')
    # ax[0].set_ylabel(r'$\lVert z \rVert^2$')
    ax[0].legend(loc=legend_loc)
    ax[1].semilogy(tt, zzhat_diff_plot, label=zzhat_diff_legend)
    ax[1].set_title('controller')
    ax[1].set_xlabel('time $t$')
    # ax[1].set_ylabel(r'$\lVert \hat{z} \rVert^2$')
    ax[1].legend(loc=legend_loc)

    if savepath is not None:
        fig.tight_layout()
        plt.savefig(savepath + '.pgf') # save as pgf
        plt.savefig(savepath + '.png') # save as png
        print(f'figure saved under savepath {savepath} (as pgf and png)')

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()

    return

# controller
def plot_outputs(
        output1: jnp.ndarray,
        output2: jnp.ndarray,
        tt: jnp.ndarray,
        output1_name: str = r'\hat{y}',
        output2_name: str = r'\overline{\hat{y}}',
        title: str = None,
        savepath: str = None,
        ):

    mpl_settings(
        figsize=(5.5,2),
        fontsize=18,
        )

    noutput = output1.shape[1]

    fig, ax = plt.subplots()

    if noutput > 1:
        output1_plot = jnp.linalg.norm(output1, axis=1)**2
        output1_legend = f'$\\lVert {output1_name}(t) \\rVert^2$'
        output2_plot = jnp.linalg.norm(output2, axis=1)**2
        output2_legend = f'$\\lVert {output2_name}(t) \\rVert^2$'
    else:
        output1_plot = output1
        output1_legend = f'${output1_name}(t)$'
        output2_plot = output2
        output2_legend = f'${output2_name}(t)$'

    ax.plot(tt, output1_plot, label=output1_legend)
    ax.plot(tt, output2_plot, label=output2_legend)
    ax.set_xlabel('time $t$')
    ax.legend()

    if savepath is not None:
        fig.tight_layout()
        plt.savefig(savepath + '.pgf') # save as pgf
        plt.savefig(savepath + '.png') # save as png
        print(f'figure saved under savepath {savepath} (as pgf and png)')

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()

    mpl_settings(
        fontsize=18,
        )

    return



if __name__ == '__main__':

    # enable double precision
    jax.config.update("jax_enable_x64", True)

    from helpers.time_integration import discrete_gradient
    from helpers.other import mpl_settings
    mpl_settings()

    nsys = 5
    T = 5.0

    A = -jnp.eye(nsys)
    B = jnp.zeros((nsys,2))
    B = B.at[0,0].set(1.0)
    B = B.at[-1,-1].set(-1.0)
    hamiltonian = lambda z: 1/2 * z.T @ z # for discrete gradient method
    eta = jax.grad(hamiltonian) # for discrete gradient method
    # r = lambda v: v # for discrete gradient method
    z0 = jnp.ones((nsys,))

    def r(v):
        return v

    def ham_eta(z):
        return hamiltonian(z), eta(z)

    def control(t):
        # return jnp.zeros((t.shape[0],2))
        return jnp.array([jnp.sin(t), jnp.cos(t)])
    control = jax.vmap(control, in_axes=0, out_axes=0,) # 0 = index where time is


    ham_tuplist = []
    error_tuplist = []

    for nt in [1000, 500, 100]:

        tt = jnp.linspace(0.0, T, nt)
        dg_result = discrete_gradient(
            r,
            ham_eta,
            lambda z: B,
            tt,
            z0,
            control(tt),
            return_hamiltonian=True,
            )

        ham, errors = discrete_gradient_powerbalance(
            r,
            ham_eta,
            lambda z: B,
            tt,
            dg_result,
            control(tt),
            relative=True,
            )

        ham_tuplist += [(tt, ham, f'nt = {nt}')]
        error_tuplist += [(tt, errors, f'nt = {nt}')]

    visualize_hamiltonian(
    ham_tuplist,
        title='evolution of hamiltonian with discrete gradient method for LTI system',
        )

    visualize_errors(
        error_tuplist,
        title='relative error in energy balance of discrete gradient method for LTI system',
        )
