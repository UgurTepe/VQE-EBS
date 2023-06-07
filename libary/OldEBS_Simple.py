import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

#np.seterr(divide='ignore', invalid='ignore')

#Running mean
def mean_ith(arr):
	return np.cumsum(arr) / np.arange(1,Max_Samples+1)

#Running variance
def var_ith(arr):
	return pd.Series(arr).expanding().var(ddof=0).to_numpy()

#c_t squence standard dt = c/t^p
def ct0(var):
	ln_constant    = np.log(c1,dtype=np.float64)/T_times
	ln_vari		   = p*np.log(T_times,dtype=np.float64)/T_times
	ln_compl 	   = ln_constant+ln_vari
	return np.sqrt(2*var*ln_compl)+ (3*R*ln_compl)

#cummax 			  
def cummax(arr):
	return np.maximum.accumulate(arr)

#cummin
def cummin(arr):
	return np.minimum.accumulate(arr)
	
#Histogram of dist.
def histo(x,n_bins):
	plt.hist(x,bins=n_bins)
	plt.show()

#Plot of the bounds
def boundplot():
	plt.semilogx(T_times,upp_b_min_epsi,T_times,low_b_max_epsi)
	#plt.plot(T_times,mean_x)
	plt.xlim(right=700)
	plt.grid()
	plt.show()

#Constants
a 		   = 1
b  		   = 5
p          = 1.1
delta	   = 0.1
c0 		   = (delta*(p-1))/p
c1	       = 3/c0
#var0 	   = (a*b)/((a+b)(a+b)(a+b+1))
erw 	   = a/(a+b)
R          = 1
epsilon    = 0.1

#time stuff
Max_Samples    = np.power(10,6)
T_times 	   = np.arange(1,Max_Samples+1)

#Construction of x, x_mean and x_var
x = np.random.default_rng().uniform(0, 1, Max_Samples)
mean_x		   = mean_ith(x)
var_x     	   = var_ith(x)
ct  	       = ct0(var_x) 
mean_x_eps     = epsilon*mean_x

#Algorithm 
for i,(e1,e2) in enumerate(zip(ct,mean_x_eps)):
	if e1 <= e2:
		print(i)
		break