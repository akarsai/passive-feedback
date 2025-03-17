#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file tests the proposed discrete gradient method
# on the example of a nonlinear pendulum
#
#

if __name__ == '__main__':

    SAVEPATH = './results'

    # jax
    import jax
    import jax.numpy as jnp

    # activate double precision and debug flags
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", True)

    # controller
    from controller import Controller

    # time discretization
    from helpers.time_integration import implicit_midpoint, discrete_gradient

    # visualization
    import matplotlib.pyplot as plt
    from helpers.visualization import discrete_gradient_powerbalance, visualize_hamiltonian, visualize_errors, plot_coupled_trajectory
    from helpers.other import mpl_settings
    mpl_settings(
        fontsize=18,
        )

    # examples
    from examples.pendulum import Pendulum
    from examples.van_der_pol import VanDerPol

    # plants
    pendulum = Pendulum()
    van_der_pol = VanDerPol()

    # control
    def sine_control(t: float) -> jnp.ndarray:
        return jnp.array([jnp.sin(t)])

    sine_control = jax.vmap(sine_control, in_axes=0, out_axes=0)
    zero_control = jax.vmap(lambda t: jnp.zeros((pendulum.ncontrol,)), in_axes=0, out_axes=0)


    # parameters
    T = 10.0 # final time
    nt_discrete_gradient = 500

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



    ###  test controller
    def pendulum_ham_eta(z):
        return pendulum.hamiltonian(z), pendulum.eta(z)

    # obtain reference solution
    tt_ref = jnp.linspace(0,T,nt_ref)
    uu_ref = sine_control(tt_ref)
    # zz_ref = discrete_gradient(f_ham_eta, pendulum.g, tt_ref, pendulum.initial_state, uu_ref)
    zz_ref = implicit_midpoint(pendulum.dynamics, tt_ref, pendulum.initial_state, uu_ref)

    print(f'reference solution obtained!')

    max_errors = []

    for k in range(num_Delta_t_steps):
        nt = int(nt_array[k])

        tt = jnp.linspace(0,T,nt)
        uu = sine_control(tt)

        zz = discrete_gradient(
            pendulum.r,
            pendulum_ham_eta,
            pendulum.g,
            tt,
            pendulum.initial_state,
            uu,
            )

        # eval reference solution on coarse time gitter
        zz_ref_resampled = zz_ref[::2**(k+ref_order_smaller),:]

        diff = zz_ref_resampled - zz

        # calculate relative error
        error = jnp.linalg.norm(diff, axis=1)#/jnp.linalg.norm(zz_bdf, axis=1) # norms along axis 1, since axis 0 are the time points
        error = error/jnp.max(jnp.linalg.norm(zz_ref, axis=1))

        max_error = jnp.max(error) # Linf error
        max_errors.append(max_error)

        print(f'done for nt={nt}')

    plt.loglog(Delta_t_array, max_errors, label='discrete gradient method', marker='o', markersize=4, color=plt.cm.tab20(0))
    # plt.xticks(nt_array)

    # comparison
    c2 = max_errors[-2]/Delta_t_array[-2]**2
    secondorder = c2*Delta_t_array**2
    plt.loglog(Delta_t_array, secondorder, label='$(\Delta t)^2$', marker='o', markersize=8, linestyle='--', color=plt.cm.tab20(1), zorder=0)

    plt.xlabel('$\Delta t$')
    plt.ylabel('relative error')
    # plt.title(f'{type} simulation for $\\dot{{z}} = Az + Bu$ with {method} method')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{SAVEPATH}/pendulum_convergence.png')
    plt.savefig(f'{SAVEPATH}/pendulum_convergence.pgf')
    plt.show()

    # plot energy balance
    tt_powerbalance = jnp.linspace(0,T,nt_discrete_gradient)
    uu_powerbalance = sine_control(tt_powerbalance)
    zz_ham = discrete_gradient(
        pendulum.r,
        pendulum_ham_eta,
        pendulum.g,
        tt_powerbalance,
        pendulum.initial_state,
        uu_powerbalance,
        return_hamiltonian=True,
        )
    ham, errors = discrete_gradient_powerbalance(
        pendulum.r,
        pendulum_ham_eta,
        pendulum.g,
        tt_powerbalance,
        zz_ham,
        uu_powerbalance,
        relative=True,
        )

    visualize_hamiltonian(
        [(tt_powerbalance, ham, r'$u(t) = \sin(t)$')],
        title='evolution of hamiltonian (nonlinear pendulum)',
        # ylabeltext=r'$\hat{\mathcal{H}}(\hat{z}(t))$',
        )

    visualize_errors(
        [(tt_powerbalance, errors, r'$u(t) = \sin(t)$')],
        title='relative error in power balance (nonlinear pendulum)',
        savepath=f'{SAVEPATH}/pendulum_powerbalance',
        )




    ###  test controller

    for tested_system in ['pendulum', 'van_der_pol']:

        if tested_system == 'pendulum':
            plant = pendulum
            options = {
                'ocp_max_iter': 50,
                # 'ocp_debug': False, # True,
                'simulation_nt': nt_discrete_gradient,
                }
        else:
            plant = van_der_pol
            options = {
                'pol_degree_x': 15,
                'pol_degree_y': 15,
                'ocp_max_iter': 80,
                'simulation_nt': nt_discrete_gradient,
                }

        controller = Controller(
            plant,
            options = options,
            )

        # for tested_control in ['sine_control', 'zero_control']:
        for tested_control in ['zero_control']:

            if tested_control == 'sine_control':
                control = sine_control
                label = r'\sin(t)'
            elif tested_control == 'zero_control':
                control = zero_control
                label = r'0'
                controller.controller_initial_state = jnp.ones((plant.nsys,)) # change initial condition
            else:
                raise NotImplementedError('other control types are not implemented')

            # compute controller trajectory
            controller_trajectory_and_hamiltonian = controller.simulate_controller_trajectory(
                control=control,
                title=f'controller dynamics with {tested_control} ({tested_system})',
                )

            # compute hamiltonian and power balance
            ham, errors = discrete_gradient_powerbalance(
                controller.controller_r,
                controller.controller_ham_eta,
                controller.plant.g,
                controller.simulation_tt,
                controller_trajectory_and_hamiltonian,
                control(controller.simulation_tt),
                relative=True,
                )

            # # repeat the simulation for EKF observer gain
            # compute controller trajectory
            # _, ham_ekf = controller.simulate_ekf_trajectory(
            #     control=control,
            #     title=f'EKF dynamics with {tested_control} ({tested_system})',
            #     )

            # plot energy balance for controller
            visualize_hamiltonian(
                [
                    (controller.simulation_tt, ham, f'$\\hat{{\\mathcal{{H}}}}(\\hat{{z}}(t))$'),
                    # (controller.simulation_tt, ham_ekf, f'$\\hat{{\\mathcal{{H}}}}(\\overline{{z}}(t))$'),
                    ],
                title=f'evolution of hamiltonian for controller dynamics ({tested_system})',
                ylabeltext=r'$\hat{\mathcal{H}}$',
                axis_type='semilogy' if tested_control == 'zero_control' else 'linear',
                savepath=f'{SAVEPATH}/{tested_system}_controller_hamiltonian_{tested_control}',
                )

            visualize_errors(
                [(controller.simulation_tt, errors, f'$\hat{{u}}(t) = {label}$')],
                title=f'relative error in power balance for controller dynamics ({tested_system})',
                savepath=f'{SAVEPATH}/{tested_system}_controller_powerbalance_{tested_control}',
                )
