import sys
import os
sys.path.append(os.path.abspath("./libary"))
from gates import *

operator = np.array([[1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [0, 0, -1, 1],
                     [0, 0, 1, -1]])

unitary = (1/np.sqrt(2))*np.array([[1, -1, 0, 0],
                                   [1, 1, 0, 0],
                                   [0, 0, 1, -1],
                                   [0, 0, 1, 1]])

def circuit_theta(theta1, theta2, state):
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
