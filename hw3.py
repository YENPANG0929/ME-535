'''
Homework 3 - ME 535 Winter 2024
Due Tuesday 06 February 2024
__This is an individual assignment, so the answers you submit must be your own work.__
Before you upload the file to Canvas, execute the final version of the file to make sure that it runs properly and produces your intended answers.
Insert your code to implement the functions as specified (without changing function names and/or arguments).
Do not import from libraries other than those that are already imported into the template (such as numpy, scipy).
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy

'''
1) (3 points) This question involves using a center-difference formula for the first derivative from Kutz' Table 4.2 
# to estimate the value of df/dx for the function f(x) = sin(x) at x=1.0. 

For those who do not have the text near at hand, the formula (with `x` as the independent variable and `h` as the spacing) is:
f'(x) = (f(x-2h) - 8 f(x-h) +8 f(x+h) -f(x+2h)) / (12 * h)

Your particular tasks are as follows:

- Compute the relative error in the derivative estimate for sample spacing values h = 2**(-n) for n in {2,3,...,18}.
- Create a log-log plot of the error as a function of sample spacing.
'''

def center_diff_rad1(f,x,h):
	'''
	Compute the radius 1 centered difference estimate of the first derivative of f at x
	Args:
		f: function to evaluate
		x: float value at which to estimate derivative
		h: float stepsize/spacing between sample points
	Returns:
		Float estimate of f'(x)
	'''
	df = (f(x+h)-f(x-h))/(2*h)
	return df


def center_diff_rad2(f,x,h):
	'''
	Compute the radius 2 centered difference estimate of the first derivative of f at x
	Args:
		f: function to evaluate
		x: float value at which to estimate derivative
		h: float stepsize/spacing between sample points
	Returns:
		Float estimate of f'(x)
	'''
	### INSERT CODE HERE

	df = (-f(x+2*h)+8*f(x+h)-8*f(x-h)+f(x-2*h))/(12*h)

	return df

def compute_diff_errors(f,x,emax, exact):
	'''
	Compute derivative estimate errors for an array of discretizations.
	
	Args:
		f: function to integrate
		x: float function argument
		emax: int specifying the range of spacing exponents
		exact: float value for error computation
	Returns:
		h: float array of sample spacings
		errors: float array of relative error values
	'''
	### INSERT CODE HERE
	h = np.zeros(len(range(2, emax+1)))
	errors = np.zeros(len(range(2, emax+1)))

	for n in range(2, emax+1):
		h_ = 2.**(-n)
		est = center_diff_rad2(f, x, h_)
		error = np.abs(exact - est) / exact
		h[n-2] = h_
		errors[n-2] = error

	return h, errors

def plot_errors(h, errors, problem, symbol='+'):
	plt.title("#{}: Error vs. stepsize".format(problem))
	plt.xlim(1.e-6, 1.e0)
	plt.ylim(1.e-16, 1.e0)
	plt.xlabel("sample spacing h")
	plt.ylabel("relative error")
	plt.loglog(h, errors, symbol)
	plt.grid()
	plt.show()

def p1():
	emax = 18
	x = 1.
	f = np.sin
	exact = np.cos(1.)
	h, errors = compute_diff_errors(f,x,emax,exact)
	p = plot_errors(h, errors, 1, symbol = '+')
	return h, errors

'''
2) (3 points) Identify an appropriate function in one of the imported packages (numpy or scipy) to fit the erro vs. spacing data to a power law 
to obtain an estimate of the order of the finite difference method.

Inspect the legend to estimate order of truncation error.
'''

def powlaw(x, c0, c1) :
	'''
	Specify a power law relationship (y = c0 * x**c1) to pass to the power_fit function
	'''
	return c0 * np.power(x, c1)

def power_fit(x, y):
	'''
	Use library function to compute best fit of data to power law y=a[0] * x**a[1]
	
	Args:
		x: 1D numpy array of float abscissa values
	y: 1D numpy array of float ordinate values
	Returns:
		Container storing  power law fit coefficients c[0], c[1]
	'''
	### INSERT CODE HERE

	c, _ = scipy.optimize.curve_fit(powlaw, x, y)

	return c
	
def plot_with_fit(h, errors, problem, symbol='+'):
	#a, b = scipy.optimize.curve_fit(powlaw, h, errors)
	c = power_fit(h, errors)
	vals = c[0]*np.power(h, c[1])
	plt.title("#{}: Error vs. stepsize".format(problem))
	plt.xlim(1.e-6, 1.e0)
	plt.ylim(1.e-16, 1.e0)
	plt.xlabel("sample spacing h")
	plt.ylabel("relative error")
	plt.loglog(h, errors, symbol)
	label_str = f"Best fit with slope {c[1]:.2f}"
	plt.loglog(h, vals, label=label_str)
	plt.grid()
	plt.legend()
	plt.show()


def p2(h,e):
	plot_with_fit(h, e, 2)
	

'''
3) (6 points) This question involves looking at the convergence properties 
of a well-known method for numerical integration or quadrature known as 
Simpson's 3/8 rule (described by Eq.(4.2.6c) in the text).
'''

def simpson38(f,xmin,xmax,p):
	'''
	Compute numerical quadrature value using composite Simpson's 3/8 rule
	Args:
		f: function corresponding to the integrand
		xmin, xmax: floats defining integration range
		p: int number of panels for the composition integration scheme
	Returns: float quadrature estimae
	'''
	### INSERT CODE HERE

	h = (xmax - xmin)/(3*p)
	quad = f(xmin) + 3*f(xmin + h) + 3*f(xmin + 2*h) + f(xmax)
	for i in range (3, 3*p):
		x = xmin + i*h
		if i % 3 == 0:
			quad += 2*f(x)
		else:
			quad += 3*f(x)

	return h, (3/8)*h*quad

def compute_integ_errors(f,xmin,xmax,n, exact):
	'''
	Compute the Simpson 38 quadrature errors for an array of discretizations.
	
	Args:
		f: function to integrate
		xmin, xmax: float limits of integration interval
		n: int specifying the range of number of integration panels (p = 2**j for j<n)
		exact: float value for error computation
	Returns:
		h: float array of sample spacings
		errors: float array of relative error values
	'''
	### INSERT CODE HERE

	h = np.zeros(len(range(0, n)))
	errors = np.zeros(len(range(0, n)))

	for j in range(0, n):
		p = 2**j
		est = simpson38(f, xmin, xmax, p)
		error = np.abs(exact - est[1]) / exact
		h[j] = est[0]
		errors[j] = error

	return h, errors

def p3():
	n = 16
	xmin, xmax = 0, np.pi
	f = np.sin
	exact = 2.
	h, errors = compute_integ_errors(f,xmin,xmax,n,exact)
	plot_errors(h, errors, 3, symbol = '+')
	return h, errors

'''
4) (0 points) Curve fit linear portion of quadrature data to show power law fit.
Inspect the legend to estimate order of truncation error.
'''

def p4(h,e):
	plot_with_fit(h, e, 4)

'''
5) (2 points) Find the library function for solving a linear system. Use it in your definition of lib_solve().
p5() will test your function to solve a circulant matrix problem and test for small residual.
'''

def lib_solve(A, b):
	'''
	Use a library function to solve the linear system Ax = b
	
	Args:
		A: 2D square (n x n) numpy float array
		b: 1D numpy float array of length n (RHS)
	Returns: 1D numpy float array that solves Ax = b
	'''
	# INSERT CODE HERE
	return np.linalg.solve(A, b)

def p5(n):
	A = scipy.linalg.circulant(1.*np.arange(1,n+1))
	b = np.random.randint(n, size=n)
	x = lib_solve(A, b)
	print("#5. Residual less than 1e-8? ", np.allclose(A@x,b, atol=1e-8))
	
'''
6) (3 points) Find the python library function for computing the eigenvalues of a matrix, 
and for computing the QR factorization. the p6() function will apply those functions
to a 10 x 10 symmetric matrix constructed from the digits of pi.

Compare/contrast the results of the eigenvalue library function and the output
from the QR <-> RQ iteration implemented in the qr_iter() function.
'''
def lib_eigenvalues(A):
	# INSERT CODE HERE

	return np.sort(scipy.linalg.eigvals(A))

def lib_qr(A):
	# INSERT CODE HERE

	return scipy.linalg.qr(A)

def qr_iter(A, N):
	for i in range(N):
		Q, R = lib_qr(A)
		A = R@Q
	return np.diagonal(A)

def p6():
	pi_str = "3141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067"
	pi_arr = [int(x) for x in str(pi_str)]
	pi_mat = np.reshape(pi_arr, [10,10])
	A = pi_mat+pi_mat.T
	evals = lib_eigenvalues(A)
	print("#6.")
	print("Eigenvalues from lib: \n", evals)
	N = 20
	qr_iter_vals = np.sort(qr_iter(A, N))
	print("Values from QR iteration: \n", qr_iter_vals)
	print("All close? :", np.allclose(evals, qr_iter_vals, rtol=1e-2))
	
def main():
	h, e = p1()
	n = 8
	spacing = h[:n]
	error = e[:n]
	p2(spacing, error)
	h, e = p3()
	n = 8
	spacing = h[:n]
	error = e[:n]
	p4(spacing, error)
	n = 20
	p5(n)
	p6()

if __name__ == "__main__":
	main()