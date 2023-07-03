import sys
import os
from datetime import date
import datetime
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

# Import own functions
sys.path.append(os.path.abspath("./libary"))
from gates import *
from functions import *
from Algorithms import nas_abs as bernstein
sys.path.append(os.path.abspath("./config"))
from circuit import *
# Current Date and Time for saving data
now = date.today().strftime("D%y-%m-%dT") + \
    datetime.datetime.now().strftime("%H_%M_%S")

'''
Initialization of Parameters 
'''
par1 = 0  # Initial parameters
par2 = 0
alpha = 0.25 # Gradient descent
eps_arr = np.logspace(-1,0,10)
index = int(sys.argv[1])-1
eps_alg = eps_arr[index] # SPSA
eps = 0.1
shift = 4
max_n = 100  # Max number of steps for GradDe
max_sample = 1*10**4  # Max number of samples for EBS
bound = 2*np.pi  # Interval of plot
break_flag = False
max_inside_minima = 5  # Variable to test convergence of algorithm within x% of minima

'''
Initialization of Arrays and Variables 
'''
energy = 0
est_energy = 0
variance = 0
energy_shifted_plus = 0
energy_shifted_minus = 0
grad1 = 0
grad2 = 0
N_step = 0
it_inside_minima = 0
arr_energy = []
arr_est_energy = []
arr_var = []
arr_est_var = []
arr_par1 = []
arr_par2 = []
arr_steps = []
arr_it = []
arr_ct = []


print(f'Initial parameters: Theta1 = {par1:.3f}, Theta2 = {par2:.3f}')
for i in range(max_n):
    # set states to |00>
    # Apply circuit on state
    state = circuit_theta(par1, par2, np.array([1, 0, 0, 0]))

    # SPSA method
    rnd1 = np.random.choice([-1, 1])
    rnd2 = np.random.choice([-1, 1])
    state_shift_plus = circuit_theta(
        par1 + eps*rnd1, par2 + eps*rnd2, np.array([1, 0, 0, 0]))
    state_shift_minus = circuit_theta(
        par1 - eps*rnd1, par2 - eps*rnd2, np.array([1, 0, 0, 0]))

    '''
    Loops for EBS algorithm
    '''
    # Resets EBS every outer loop iteration
    ebs = bernstein(delta=0.1, epsilon=eps_alg)
    ebs_shift_plus = bernstein(delta=0.1, epsilon=eps_alg)
    ebs_shift_minus = bernstein(delta=0.1, epsilon=eps_alg)

    # EBS for Energy(x,y)
    while ebs.cond_check():
        ebs.add_sample(np.sum(measure(state, 'zx'))+shift)
        if ebs.get_step() > max_sample:
            break
    # Saving values
    it_normal = ebs.get_step()
    est_energy = ebs.get_estimate()-shift

    energy = expected_value(state, operator)
    variance = expected_value(state, np.linalg.matrix_power(
        operator,2)) - expected_value(state, operator)**2

    # EBS for Energy(x+εΔ,y+εΔ)
    while ebs_shift_plus.cond_check():
        ebs_shift_plus.add_sample(np.sum(measure(state_shift_plus, 'zx'))+shift)
        if ebs_shift_plus.get_step() > max_sample:
            break
    energy_shifted_plus = ebs_shift_plus.get_estimate()-shift

    # EBS for Energy(x-εΔ,y-εΔ)
    while ebs_shift_minus.cond_check():
        ebs_shift_minus.add_sample(np.sum(measure(state_shift_minus, 'zx'))+shift)
        if ebs_shift_minus.get_step() > max_sample:
            break
    energy_shifted_minus = ebs_shift_minus.get_estimate()-shift

    # Save data
    arr_energy.append(energy)
    arr_est_energy.append(est_energy)
    arr_var.append(variance)
    arr_est_var.append(0)
    arr_par1.append(par1)
    arr_par2.append(par2)
    arr_steps.append(it_normal)
    arr_it.append(i)
  
    # Estimate the Gradient via SPSA method
    grad1 = (energy_shifted_plus-energy_shifted_minus) / (2*eps*rnd1)
    grad2 = (energy_shifted_plus-energy_shifted_minus) / (2*eps*rnd2)
    
    print(
        f'+---------------------------------------[i={i}]----------------------------------------+')
    print(
        f'State: | {state[0]:.5f} {state[1]:.5f} {state[2]:.5f} {state[3]:.5f} >')
    print(
        f'Parameters: Theta1={par1:.5f}, Theta2={par2:.5f}')
    print(
        f'Estimated Energy={est_energy:.5f}, Analytical Energy={energy:.5f}')
    print(
        f'Analytical Variance={variance:.5f}')
    print(
        f'Gradient: ∇(Theta1)={grad1:.5f}, ∇(Theta2={grad2:.5f})')
    print(f'Samples = {it_normal}')
    # Updating the Parameters via Gradient Descent method
    par1 -= alpha*grad1
    par2 -= alpha*grad2

    # Ensures convergence of the algorithm
    if np.abs(est_energy + 2) <= 0.2:
        print(
            '******************************************************************')
        print('Aborted: within 0.2 of minima')
        print(f'At i = {i}')
        print(
            '******************************************************************')
        print(f'Energy = {est_energy}')
        N_step = i
        it_inside_minima += 1
        if it_inside_minima >= max_inside_minima:
            break
        else:
            continue


'''
Visualization of Parameters and Energy/Variance
'''
plot_landscape(operator, np.array([1, 0, 0, 0]), (-bound, bound),
               (-bound, bound), 200, (arr_par1, arr_par2), var_on=False)
plt.plot(arr_par1,arr_par2,'r-')
plt.plot(arr_par1,arr_par2,'k.')
plt.plot(arr_par1[0],arr_par2[0],'x',label = 'Start')
plt.plot(arr_par1[-1],arr_par2[-1],'o',label = 'End')
plt.legend()

#folder = f'output/shift_comp/VQE_a{alpha:.3f}_e{eps:.3f}/'
folder = f'output/nas_test_eps/eps{eps_alg:.3f}/'
os.makedirs(folder,exist_ok = True)
plt.title(rf'$\epsilon={eps}$, $\alpha = {alpha}$, Num of Steps$={N_step}$')
plt.savefig(folder+f'fig'+now+'.png')
np.savetxt(folder+f'data'+now+'.txt', (arr_it,arr_par1,arr_par2, arr_energy, arr_var, arr_est_energy, arr_est_var, arr_steps), delimiter=',')
#plt.show()