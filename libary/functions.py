import numpy as np
from numpy import linalg as LA
from gates import *
import sys
import os
sys.path.append(os.path.abspath("./config"))
from circuit import *
import matplotlib.pyplot as plt

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

def expected_value(state, operator):
    state_dagger = state.conj()
    state = operator@state
    result = np.inner(state_dagger, state)
    return result.real

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
