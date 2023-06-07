from gates import *
import numpy as np
from numpy import linalg as LA
from collections import deque

class qmstate:
    def __init__(self, n=2, flag_random = False):
        assert n == 1 or n == 2, "n must be 1 or 2"
        self.n = n
        self.state = np.array([0,0,0,0])
        # Create state 
        if flag_random == False:
            self.state = np.zeros(2**self.n, dtype=np.int64)
            self.state[0] = 1

        if flag_random == True:
            alpha = complex(np.random.uniform(-1,1),np.random.uniform(-1,1))
            beta  = complex(np.random.uniform(-1,1),np.random.uniform(-1,1))
            gamma = complex(np.random.uniform(-1,1),np.random.uniform(-1,1))
            delta = complex(np.random.uniform(-1,1),np.random.uniform(-1,1))

            if n == 2:   
                norm = np.sqrt(np.abs(alpha)**2 + np.abs(beta)**2+ np.abs(gamma)**2+ np.abs(delta)**2)
                self.state = np.array([alpha/norm,beta/norm,gamma/norm,delta/norm])
            else:
                norm = np.sqrt(np.abs(alpha)**2 + np.abs(beta)**2)
                self.state = np.array([alpha/norm,beta/norm])

    def set_state(self,state):
        self.state = state

    def __str__(self):
        if self.n == 2:
            return f"|{self.state[0]},{self.state[1]},{self.state[2]},{self.state[3]}>"
        else:
            return f"|{self.state[0]},{self.state[1]}>"

    def __matmul___(self,A):
        self.state =  np.matmul(A,self.state)

    # def gate(self, operator, parameter = False):
    #     if parameter:
    #         assert (operator(parameter).shape[0] == (self.state.shape[0])
    #                 ), "Dimensions of gate and state do not match"
    #         self.state = operator(parameter)@self.state
    #     else:
    #         assert (operator().shape[0] == (self.state.shape[0])
    #                 ), "Dimensions of gate and state do not match"
    #         self.state = operator()@self.state

    def gate(self,operator):
        self.state = operator @ self.state 

    def measure(self,flag_basis = 'z'):
        assert len(self.state) <= 4, "Shape of state is too big. Only 1 or 2 qubit states are allowed"
        if LA.norm(self.state) != 1:
            self.normalize()

        # Measurement in z bais
        if flag_basis == 'z':
            prob_state = (np.absolute(self.state)**2).flatten()
            # Chooses either |00>,|10>,|01> or |11> according to their respective probabilities
            a = np.random.choice(len(prob_state), p=prob_state)
            x = int(len(prob_state)/2)
            b = np.binary_repr(a, width=x)
            c = np.array([int(x) for x in b])
            return -2*c+1

        # Measurement in x basis
        if flag_basis == 'x':
            if len(self.state) == 4:
                gate = np.kron(h_gate(),h_gate())
            elif len(self.state) == 2:
                gate = h_gate()
            prob_state = (np.absolute(gate@self.state)**2).flatten()
            #Chooses either |00>,|10>,|01> or |11> according to their respective probabilities
            a = np.random.choice(len(prob_state),p=prob_state)
            x = int(len(prob_state)/2)
            b = np.binary_repr(a,width = x)
            c = np.array([int(x) for x in b])
            return -2*c+1

        # Measurment in y basis
        if flag_basis == 'y':
            if len(self.state) == 4:
                gate = np.kron(s_dagger_gate()@h_gate(),s_dagger_gate()@h_gate())
            elif len(self.state) == 2:
                gate = s_dagger_gate()@h_gate()
            prob_state = (np.absolute(gate@self.state)**2).flatten()
            #Chooses either |00>,|10>,|01> or |11> according to their respective probabilities
            a = np.random.choice(len(prob_state),p=prob_state)
            x = int(len(prob_state)/2)
            b = np.binary_repr(a,width = x)
            c = np.array([int(x) for x in b])
            return -2*c+1

        # Measurment in zx basis
        if flag_basis == 'zx':
            return np.array([self.measure()[0],self.measure('x')[1]])

        # Measurement in zx basis
        if flag_basis == 'xz':
            return np.array([self.measure('x')[0],self.measure()[1]])

    def normalize(self):
        norm =  LA.norm(self.state)
        self.state = self.state / norm

    def is_normalized(self):
        norm = np.sqrt(np.sum(self.state**2))
        if norm == 1:
            print(True)
        else:
            print(False) 
            
    def expected_value(self, operator):
        assert (operator.shape[0] == (self.state.shape[0])
                ), "Dimensions of gate and state do not match"
        state_dagger = self.state.conj()
        state = operator@state
        result = np.inner(state_dagger, state)
        return result.real

    def get_state(self):
        return self.state

class qmcircuit:
    def __init__(self):
        self.state = np.zeros(2)
        self.state[0] = 1
        self.circuit = deque()
        self.circuit_complete = np.eye(2)

    # Runs the circuit
    def run_circuit(self):
        while self.circuit.empty() != True:
            self.circuit.pop_gate()

    # Initalize state
    def set_state(self,state):
        self.state = state

    # Fills gates left to right
    def build_circuit(self,*gates):
        self.circuit.extend(gates)

    def set_circuit(self,operator, *parameters):
        if len(parameters) == 0:
            self.circuit_complete = operator()
        else:
            self.circuit_complete = operator(parameters)

    def eval_circuit(self):
        self.state = self.circuit_complete @ self.state

    # evaluates a gate (left to right)
    def pop_gate(self):
        self.state = self.circuit.popleft() @ self.state

    # Returns gate
    def get_circuit(self):
        return self.circuit

    # Retruns state
    def get_state(self):
        return self.state
