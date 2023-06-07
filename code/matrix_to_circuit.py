import sys
import os
import numpy as np
from numpy import linalg as LA
import quantum_decomp as qd
import qiskit.quantum_info as qi
# Import own functions
sys.path.append(os.path.abspath("./libary"))
from gates import *
from functions import *
from bernstein_class import Bernstein as bernstein
sys.path.append(os.path.abspath("./config"))
from circuit import *

Operator = np.array([[1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [0, 0, -1, 1],
                    [0, 0, 1, -1]])

lmb = np.array([[2, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, -2]])

lmb, U = np.linalg.eig(Operator)
lmb = np.diag(lmb)

# Input Matrix
print("Input Matrix")
print(Operator)

#  Eigenvalue decomp
print("U: Eigenvec")
print(U)

print("inv(U): Eigenvec")
invU = np.linalg.inv(U)
print(invU)

print("Λ: Eigenvalues:")
print(lmb)

print('U*Λ*inv(U)')
print(U@lmb@invU)

# Output Gate
print("Output Gate")
gate_decomp = qd.matrix_to_qiskit_circuit(U)
print(gate_decomp)
op = qi.Operator(gate_decomp)

print('Test')
print(1/(np.sqrt(2))*invU@np.array([1,1,0,0]))
