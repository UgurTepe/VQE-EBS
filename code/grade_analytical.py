import numpy as np
from numpy import linalg as LA
from gates import *
import matplotlib.pyplot as plt
from tqdm import trange
from tqdm import tqdm
import datetime
from datetime import date
now = date.today().strftime("%y-%m-%dT") + \
    datetime.datetime.now().strftime("%H_%M_%S")

# The operator on which the circuit below
# is build on
operator = np.array([[1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [0, 0, -1, 1],
                     [0, 0, 1, -1]])


def circuit_theta(theta1, theta2, state):
    '''
    H = (Z ⊗1) + (1 ⊗ X)
    ALLES IN DEM BILD MUSS INVERTIERT SEIN WEIL QISKIT CRAZY IST
          ┌───────┐┌──────────┐┌────────┐     ┌───────┐┌──────────┐┌────────┐
q_0: ─────┤ Rz(π) ├┤ Ry(-π/2) ├┤ Rz(-π) ├─────┤ Rz(π) ├┤ Ry(-π/2) ├┤ Rz(-π) ├
     ┌───┐└───┬───┘└────┬─────┘└───┬────┘┌───┐└───┬───┘└────┬─────┘└───┬────┘
q_1: ┤ X ├────■─────────■──────────■─────┤ X ├────■─────────■──────────■─────
     └───┘                               └───┘
Sec:  1                 2                 3               4
    '''

    # Pre rotation
    gate_01 = np.kron(ry_gate(theta1), ry_gate(theta2))
    state = gate_01@state
    # Section1
    gate_1 = np.kron(x_gate(), i_gate())
    state = gate_1@state
    # Section2
    gate_21 = cu_gate(rz_gate(np.pi))
    state = gate_21@state
    gate_22 = cu_gate(ry_gate(-np.pi*0.5))
    state = gate_22@state
    gate_23 = cu_gate(rz_gate(-np.pi))
    state = gate_23@state
    # Section3
    gate_31 = np.kron(x_gate(), i_gate())
    state = gate_31@state
    # Section4
    gate_41 = cu_gate(rz_gate(np.pi))
    state = gate_41@state
    gate_42 = cu_gate(ry_gate(-np.pi*0.5))
    state = gate_42@state
    gate_43 = cu_gate(rz_gate(-np.pi))
    state = gate_43@state

    return state


def expected_value(state, operator):
    state_dagger = state.conj()
    state = operator@state
    result = np.inner(state_dagger, state)
    return result.real


def measure(state, flag_basis='z'):
    assert len(
        state) <= 4, "Shape of state is too big. Only 1 or 2 qubit states are allowed"
    if LA.norm(state) != 1:
        state = normalize(state)

    # Measurement in z bais
    if flag_basis == 'z':
        prob_state = (np.absolute(state)**2).flatten()
        # Chooses either |00>,|10>,|01> or |11> according to their respective probabilities
        a = np.random.choice(len(prob_state), p=prob_state)
        x = int(len(prob_state)/2)
        b = np.binary_repr(a, width=x)
        c = np.array([int(x) for x in b])
        return -2*c+1

    # Measurement in x basis
    if flag_basis == 'x':
        if len(state) == 4:
            gate = np.kron(h_gate(), h_gate())
        elif len(state) == 2:
            gate = h_gate()
        prob_state = (np.absolute(gate@state)**2).flatten()
        # Chooses either |00>,|10>,|01> or |11> according to their respective probabilities
        a = np.random.choice(len(prob_state), p=prob_state)
        x = int(len(prob_state)/2)
        b = np.binary_repr(a, width=x)
        c = np.array([int(x) for x in b])
        return -2*c+1

    # Measurment in y basis
    if flag_basis == 'y':
        if len(state) == 4:
            gate = np.kron(s_dagger_gate()@h_gate(), s_dagger_gate()@h_gate())
        elif len(state) == 2:
            gate = s_dagger_gate()@h_gate()
        prob_state = (np.absolute(gate@state)**2).flatten()
        # Chooses either |00>,|10>,|01> or |11> according to their respective probabilities
        a = np.random.choice(len(prob_state), p=prob_state)
        x = int(len(prob_state)/2)
        b = np.binary_repr(a, width=x)
        c = np.array([int(x) for x in b])
        return -2*c+1

    # Measurment in zx basis
    if flag_basis == 'zx':
        return np.array([measure(state)[0], measure(state, 'x')[1]])

    # Measurement in zx basis
    if flag_basis == 'xz':
        return np.array([measure(state, 'x')[0], measure(state)[1]])


def normalize(state):

    norm = LA.norm(state)
    return state / norm


def plot_landscape(input_operator, input_state, x_limits, y_limits, grid_points, path, var_on=False):
    N = 30
    max_level = 2
    min_level = -2
    step_level = 0.25
    operator = input_operator
    state = input_state
    x_lower, x_upper = x_limits
    theta_x = np.linspace(x_lower, x_upper, grid_points)
    max_x = len(theta_x)
    y_lower, y_upper = y_limits
    theta_y = np.linspace(y_lower, y_upper, grid_points)
    max_y = len(theta_y)
    values = np.empty(shape=(max_x, max_y))
    values_var = np.empty(shape=(max_x, max_y))
    x_p, y_p = path

    if var_on == True:
        for i, x in enumerate(theta_x):
            for k, y in enumerate(theta_y):
                values[i][k] = expected_value(
                    circuit_theta(x, y, state), operator)
                values_var[i][k] = expected_value(circuit_theta(x, y, state),
                                                  np.linalg.matrix_power(operator, 2)) - expected_value(circuit_theta(x, y, state), operator)**2
    else:
        for i, x in enumerate(theta_x):
            for k, y in enumerate(theta_y):
                values[i][k] = expected_value(
                    circuit_theta(x, y, state), operator)

    if var_on == True:
        # Energy Plot
        plt.subplot(1, 2, 1).set_title('Energy')
        plt.contourf(theta_x, theta_y, values, N, cmap="viridis")
        plt.plot(x_p[1:-1], y_p[1:-1], 'kX')
        plt.plot(x_p, y_p, 'r--')
        plt.plot(x_p[0], y_p[0], 'H', label='Start')
        plt.plot(x_p[-1], y_p[-1], '*',  label='Ende')

        # Variance Plot
        plt.subplot(1, 2, 2).set_title('Variance')
        plt.contourf(theta_x, theta_y, values_var, N, cmap="viridis")
        plt.plot(x_p[1:-1], y_p[1:-1], 'kX')
        plt.plot(x_p, y_p, 'r--')
        plt.plot(x_p[0], y_p[0], 'H', label='Start')
        plt.plot(x_p[-1], y_p[-1], '*',  label='Ende')

        # Stuff
        plt.axis('auto')
        plt.colorbar()
        plt.legend()
        # plt.show()
    else:
        plt.contourf(theta_x, theta_y, values, levels=np.arange(
            min_level, max_level + step_level, step_level), cmap="viridis")
        plt.colorbar()
        plt.title('Energy')
        plt.axis('scaled')
        plt.plot(x_p[1:-1], y_p[1:-1], 'kX')
        plt.plot(x_p, y_p, 'r--')
        plt.plot(x_p[0], y_p[0], 'H', label='Start')
        plt.plot(x_p[-1], y_p[-1], '*',  label='Ende')
        plt.text(
            x_p[-1], y_p[-1], f' { expected_value(circuit_theta(x_p[-1], y_p[-1], state), operator):1.3f}')
        plt.legend()
        # plt.contour([0],[0],[0],1)
        # plt.show()


# Initalize States and Bernstein instance
state = np.array([1, 0, 0, 0])
state_shift_par1 = state
state_shift_par2 = state

# Random initial location and mu
th1, th2 = np.random.uniform(-np.pi, np.pi, 2)
mu_par = 0.25
mu_grad = 3*10**(-6)
max_n = 1000

# Arrays initalisation
energy = 0
est_energy = 0
variance = 0
par1 = th1
par2 = th2
energy_shifted_par1 = 0
energy_shifted_par2 = 0
grad1 = 0
grad2 = 0
arr_energy = []
arr_par1 = []
arr_par2 = []

break_flag = False
break_loop = 0
N_step = 0


print(f'Initial parameters: Theta1 = {par1:.3f}, Theta2 = {par2:.3f}')
for i in trange(max_n, desc="Loop Iteration"):
    #mu_grad *= 0.999

    energy = expected_value(
        circuit_theta(par1, par2, np.array([1, 0, 0, 0])), operator)
    variance = expected_value(circuit_theta(par1, par2, np.array([1, 0, 0, 0])),
                              np.linalg.matrix_power(operator, 2)) - expected_value(circuit_theta(par1, par2, np.array([1, 0, 0, 0])), operator)**2

    energy_shifted_par1 = expected_value(
        circuit_theta(par1 + mu_grad, par2,  np.array([1, 0, 0, 0])), operator)

    energy_shifted_par2 = expected_value(
        circuit_theta(par1, par2 + mu_grad, np.array([1, 0, 0, 0])), operator)

    # Save data
    arr_energy.append(energy)
    arr_par1.append(par1)
    arr_par2.append(par2)

    # Calculate the gradient
    grad1 = np.divide((energy_shifted_par1-energy), mu_grad)
    grad2 = np.divide((energy_shifted_par2-energy), mu_grad)

    tqdm.write(
        '+----------------------------------------------------------------+')
    tqdm.write(
        f'Expected Energy: {energy:.5f}')
    tqdm.write(
        f'Variance {variance:.3f}')
    tqdm.write(
        f'Current Parameters: Theta1={par1:.3f}, Theta2={par2:.3f}, mu = {mu_par}')
    tqdm.write(
        f'Current Gradient: ∇(Theta1)={grad1:.3f}, ∇(Theta2={grad2:.3f})')

    # Updating the parameters
    par1 -= mu_par*grad1
    par2 -= mu_par*grad2
    # par1 %= 2*np.pi
    # par2 %= 2*np.pi

    # Abort if close to wanted minimum (-2 in this case)
    if i > 1 and np.abs(energy + 2) <= 0.1:
        if break_flag == True and break_loop >= 5:
            tqdm.write(
                '******************************************************************')
            tqdm.write('Aborted: within 0.2 of minima')
            tqdm.write(f'At i = {i}')
            tqdm.write(
                '******************************************************************')
            tqdm.write(f'Energy = {energy}')
            N_step = i
            break
        else:
            break_loop += 1
            break_flag = True

bound = 2*np.pi

while par1 > bound or par2 > bound:
    bound *= 2

plot_landscape(operator, np.array([1, 0, 0, 0]),
               (-bound, bound), (-bound, bound), 200, (arr_par1, arr_par2), var_on=False)

plt.title(f'$\mu_g={mu_grad}$, $\mu_p = {mu_par}$, #Iter$={N_step}$')
# plt.savefig('output/plot_'+now+'.png')
plt.show()
# Save data
# np.savetxt('output/simdata_'+now+'.txt',(par1,par2,energies),fmt='%1.6f',delimiter=',')
