'''
Homework 5 - ME 535 Winter 2024

__This is an individual assignment, so the answers you submit must be your own work.__
Before you upload the file to Canvas, execute the final version of the file to make sure that it runs properly and produces your intended answers.
Insert your code to implement the functions as specified (without changing function names and/or arguments).
Do not import from libraries other than those that are already imported into the template (such as numpy).
'''

import numpy as np
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.pyplot as plt
from simple_plot import *
# Some additional statements for Problem 3
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import linprog
plt.rcParams['figure.figsize'] = [12, 12]
plt.rcParams.update({'font.size': 18})


def initialize_1(u,h):
	'''
	Initialize the array to account for BCs and geometry of problem 1 of Homework 4.

	Args:
		u: 2D numpy array corresponding to values sampled on a regular 2D grid
		h: float grid-spacing

	Returns:
		2D numpy array corresponding to the input with boundary values on the boundary/exterior or the triangular domain.
	'''
	# INSERT CODE HERE

	m, n = u.shape
	for i in range(m):
		for j in range(n):
			if i == 0:
				u[i, j] = j*h
			if j == 0:
				u[i, j] = (i*h)**2
			if i+j >= m-1:
				u[i, j] = 1
			
	return u

def update(u):
	'''
	Update values within the triangular domain based on the 5-point averaging stencil

	Args:
		u: 2D numpy array of input values (with appropriate boundary conditions outside the triangular domain)
	Returns:
		2D numpy array of with values updated within the triangular domain
	'''
	# INSERT CODE HERE

	for i in range(1, u.shape[0]):
		for j in range(1, u.shape[1]):
			if i+j < u.shape[0]-1:
				u[i, j] = 1/4*(u[(i-1), j] + u[(i+1), j] + u[i, (j-1)] + u[i, (j+1)])

	return u

def iter_solve(u, tol, max_iters):
	'''
	Solve the Laplace equation by iterating with 5-point averaging stencil (by calling update() )   
	
	Args:
		u: 2D numpy array to update
		tol: float maximal allowable change in an updated value
		max_iters: int limit on number of update iterations

	Return:
		2D numpy array with final solution
		iter_count: number of iterations to achieve specified tolerance
	'''
	# INSERT CODE HERE

	for k in range(1, max_iters+1):
		u_k = update(u.copy())
		if np.max(np.abs(u_k - u)) < tol:
			break
		u = u_k
		k += 1

	return u, k

def p1():
	n = 10
	h = 1/(n-1)
	u = np.zeros([n,n])
	v = initialize_1(u,h)
	# print(v)
	# v1 = update(v)
	np.set_printoptions(precision=3)
	v2, iters = iter_solve(v, .01, 30)
	print(v2, iters)
	xvals = np.linspace(0,1,n)
	yvals = np.linspace(0,1,n)
	arraycontourplot(xvals, yvals, np.flipud(v2), levels = [0.2,0.4,0.6,0.8, 0.99])

def displacement(xmin,xmax,nx):
	'''
	Initial displacement function for problem 2

	Args:
		nx: int number of positions on uniform 1d grid
	Returns:
		1d numpy float array of displacements
	'''
	x = np.linspace(xmin,xmax,nx)
	L = xmax-xmin
	d = np.maximum(0, 1 - np.abs(10*x/(L/2)))
	return d

def initialize_2(f,xmin,xmax,nx,nt):
	'''
	Set up the solution array and establish initial conditions

	Args:
		f: function that defines initial displacement
		nt: int number of time steps
		xmin,xmax: float bounds of spatial domain
		nx: int number of grid points along x-axis

	Returns:
		nt x nx float numpy array 
		with initial displacement stored for first two time steps
	'''
	u = np.zeros([nt,nx])
	disp = f(xmin,xmax,nx) #compute initial displacement
	u[0,:] = disp #store initial displacement for time step 0
	# store copy of initial displacement for zero initial velocity
	u[1,:] = disp 
	return u

def single_step(x0,x1,dx,dt):
	'''
	Compute a single update step for central difference 
	discretization of wave equation.

	Args:
		x0: 1D numpy float array of previous displacements
		x1: 1D numpy float array of current displacements
		dx: float grid-spacing
		dt: float timestep
	Returns:
		x2: 1d array of updated displacements
	'''
	# INSERT CODE HERE

	x2 = np.array(x1)

	for i in range(1, x1.shape[0]-1):
		x2[i] = 2*x1[i] - x0[i] + ((dt/dx)**2)*(x1[i-1] - 2*x1[i] + x1[i+1])

	return x2

def step_solve(f,xmin,xmax,nx,tmin,tmax,r):
	'''
	Compute the solution array for problem 2

	Args:
		f: function describing initial displacement
		xmin,xmax: float bounds on spatial interval
		nx: int number of grid points across spatial domain
		tmin,tmax: float bounds on time interval
		r: float ratio of time step to grid spacing
	Retuns:
		2D float numpy array of computed solution values

	'''
	# INSERT CODE HERE
	dx = (xmax-xmin)/nx
	dt = r*dx
	nt = int((tmax-tmin)//dt)
	u = initialize_2(f, xmin, xmax, nx, nt)
	for i in range(2, u.shape[0]):
		u[i] = single_step(u[i-2], u[i-1], dx, dt)

	return u

def p2():
	xmin,xmax = -100,100
	nx = 501
	dx = (xmax-xmin)/nx
	tmin,tmax = 0,35.
	ratio = [0.999, 1.001]
	stability = [True, False] 
	for i in range(len(ratio)):
		r = ratio[i]
		dt = r*dx
		nt = int((tmax-tmin)//dt)
		u = step_solve(displacement,xmin,xmax,nx,tmin,tmax,r)
		x = np.linspace(xmin,xmax,nx)
		t = tmin + dt*np.arange(nt)
		# Uncomment line below for contourplot
		# arraycontourplot(x,t,u.T)

		#Create surface plot of solution
		X,T = np.meshgrid(x,t)
		fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
		surf = ax.plot_surface(X, T, u, cmap=cm.coolwarm,
							linewidth=0, antialiased=False)

		# Customize the z axis.
		ax.set_zlim(-1.01, 1.01)
		ax.zaxis.set_major_formatter('{x:.02f}')
		# Add a color bar which maps values to colors.
		fig.colorbar(surf, shrink=0.5, aspect=5)
		plt.title(f'Numerical solution appears stable for r={r}? {stability[i]}')
		plt.show()
		
###
#3
###
		
# Reference code
		
def rhsHeat(uhat_ri,t,kappa,a):
	N = uhat_ri.shape[0] // 2
	uhat = uhat_ri[:N] + (1j) * uhat_ri[N:]
	d_uhat = -a**2 * (np.power(kappa,2)) * uhat
	d_uhat_ri = np.concatenate((d_uhat.real,d_uhat.imag)).astype('float64')
	return d_uhat_ri

def rhsHeat_ivp(t, uhat_ri,kappa,a):
	'''
	Updated rate function compatible with solve_ivp. 
	'''
	# INSERT CODE HERE

	N = uhat_ri.shape[0] // 2
	uhat = uhat_ri[:N] + (1j) * uhat_ri[N:]
	d_uhat = -a**2 * (np.power(kappa,2)) * uhat
	d_uhat_ri = np.concatenate((d_uhat.real,d_uhat.imag)).astype('float64')

	return d_uhat_ri.T

def rhsWave(uhat_ri,t,kappa,c):
	# Rate function for reference solution with odeint
	N = uhat_ri.shape[0] // 2
	uhat = uhat_ri[:N] + (1j) * uhat_ri[N:]
	d_uhat = -c * (1j*kappa) * uhat
	d_uhat_ri = np.concatenate((d_uhat.real,d_uhat.imag)).astype('float64')
	return d_uhat_ri

def rhsWave_ivp(t, uhat_ri,kappa,c):
	# Rate function for updated solver with solve_ivp
	# INSERT CODE HERE
	
	N = uhat_ri.shape[0] // 2
	uhat = uhat_ri[:N] + (1j) * uhat_ri[N:]
	d_uhat = -c * (1j*kappa) * uhat
	d_uhat_ri = np.concatenate((d_uhat.real,d_uhat.imag)).astype('float64')

	return d_uhat_ri.T

# REFERENCE CODE
def fft_odeint(rhs, a):
	# Parameter a is Thermal diffusivity constant or wave speed
	#a = 1    # Thermal diffusivity constant
	L = 100  # Length of domain
	#c = 2.5 # Wavespeed
	N = 500 # Number of discretization points
	dx = L/N
	x = np.arange(-L/2,L/2,dx) # Define x domain
	# Define discrete wavenumbers
	kappa = 2*np.pi*np.fft.fftfreq(N, d=dx)
	# Initial condition
	u0 = np.zeros_like(x)
	u0[int((L/2 - L/10)/dx):int((L/2 + L/10)/dx)] = 1
	u0hat = np.fft.fft(u0)
	# SciPy's odeint function doesn't play well with complex numbers, so we recast 
	# the state u0hat from an N-element complex vector to a 2N-element real vector
	u0hat_ri = np.concatenate((u0hat.real,u0hat.imag))
	
	# Simulate in Fourier frequency domain
	dt = 0.1
	t = np.arange(0,10,dt)
	# Call the old odeint solver to solve for transformed values
	uhat_ri = odeint(rhs, u0hat_ri, t, args=(kappa,a))
	# Re-assemble complex values
	uhat = uhat_ri[:,:N] + (1j) * uhat_ri[:,N:]

	u = np.zeros_like(uhat)
	# For each timestep, invert the transform to get spatial dependence
	for k in range(len(t)):
		u[k,:] = np.fft.ifft(uhat[k,:])
	# Keep the real part (and ignore small imaginary components)
	u = u.real    
		
	# Image plot
	plt.figure()
	plt.imshow(u, aspect=8)
	#plt.axis('off')
	plt.xlabel('X')
	plt.ylabel('Time')
	plt.title(f'Reference solution using old odesolve\n {rhs}')
	plt.colorbar()
	plt.show()

def fft_ivp_solve(rhs,a):
	# UPDATED SOLVER USING solve_ivp
	# Parameter a is Thermal diffusivity constant or wave speed
	L = 100  # Length of domain
	N = 500 # Number of discretization points
	dx = L/N
	# INSERT CODE HERE: NAME YOUR SOLUTION u FOR PLOTTING

	x = np.arange(-L/2,L/2,dx) # Define x domain
	# Define discrete wavenumbers
	kappa = 2*np.pi*np.fft.fftfreq(N, d=dx)
	# Initial condition
	u0 = np.zeros_like(x)
	u0[int((L/2 - L/10)/dx):int((L/2 + L/10)/dx)] = 1
	u0hat = np.fft.fft(u0)
	# SciPy's odeint function doesn't play well with complex numbers, so we recast 
	# the state u0hat from an N-element complex vector to a 2N-element real vector
	u0hat_ri = np.concatenate((u0hat.real,u0hat.imag))
	
	# Simulate in Fourier frequency domain
	dt = 0.1
	t = np.arange(0,10,dt)
	uhat_ri = solve_ivp(rhs, (t[0], t[-1]), u0hat_ri, args=(kappa,a))

	u = np.zeros((len(uhat_ri.t), N))
	for k in range(len(uhat_ri.t)):
		u[k, :] = np.fft.ifft(uhat_ri.y[:, k][:N] + 1j * uhat_ri.y[:, k][N:])

	u = u.real

	# Image plot
	plt.figure()
	plt.imshow(u, aspect=8)
	#plt.axis('off')
	plt.xlabel('X')
	plt.ylabel('Time')
	plt.title(f'Solution using updated solve_ivp\n {rhs}')
	plt.colorbar()
	plt.show()
	
	 
def p3():
	a = 1
	c = 4
	fft_odeint(rhsHeat,1)
	fft_odeint(rhsWave,4.)
	fft_ivp_solve(rhsHeat_ivp,1)
	fft_ivp_solve(rhsWave_ivp,4.)
	 
####
#4
####
	 
def my_linprog(equality=True):
	'''
	Refer to PDF for problem description
	'''
	# INSERT CODE HERE

	obj = [-1, -2]

	lhs_ineq = [[ 1,  1], [-4,  5], [ 1, -2]]
	rhs_ineq = [9, 10, 2]

	lhs_eq = [[-1, 5]]
	rhs_eq = [15]

	bnd = [(0, float("inf")), (0, float("inf"))]

	if equality == True:
		opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,
	            	A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd,
 	            	method="highs")
	else:
		opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, bounds=bnd,
 	            	method="highs")

	return opt

def p4():
	np.set_printoptions(precision=2)
	opt = my_linprog(equality=True)
	print("With equality contraint:")
	print("Optimum ", opt.fun, "at (x,y) = ", opt.x)

	opt = my_linprog(equality=False)
	print("Without equality contraint:")
	print("Optimum ", opt.fun, "at (x,y) = ", opt.x)

if __name__ == '__main__':
	p1()
	p2()
	p3()
	p4()