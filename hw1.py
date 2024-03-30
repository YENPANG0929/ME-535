# HW1 - ME535 W24

###
# 1 
###
""" 
1)  Consider a 1-byte floating-point system with 1 sign bit, 4 precision bits (for the mantissa), and 3 exponent bits corresponding to: $\beta=2, p=4, e \in \{-4,-2,-1,0,1,2,3\}$.
Fill in code in the template below to implement a function that computes the represented number corresponding to a specific set of input values. 
Assume that all input values are of the appropriate type (i.e., `d` is an array of integers and the other arguments are integers). 
"""

import numpy as np
import matplotlib.pyplot as plt

def digits_to_num(s, d, e, beta=2):
    """
    Compute the numerical value corresponding to floating point parameter values

    Args:
        s: 0 for positive, 1 for negative
        d: array of integer digits in range(0,beta)
        e: integer exponent value
    Named args:
        base: positive integer, default value 2

    Returns:
        float val: value of represented number
    """
    # INSERT YOUR CODE HERE
    
    val = 0
    d_val = 0
    
    for i in range(len(d)):
        d_val += d[i] / beta**i
    
    val = (-1)**s * d_val * beta**e

    return val

""" 
Test your function definition by entering `(digits_to_num(0, [1,1,1,1], 3)` in a code cell and evaluating. 
This should produce the largest number the system can represent. What should this largest value be? 
Check that your function produces the appropriate value. 
"""
def p1():
	s = 0
	d = np.ones(4)
	e_min, e_max = -4, 3
	print("Largest absolute value is: ", digits_to_num(s, d, e_max))

	d = np.array([1,0,0,0])
	print("Smallest absolute value is: ", digits_to_num(s, d, e_min))


### Call the function with some other digit arrays, exponents, and signs to verify

###
# 2) Compute the gamut of representable numbers in the example system above.
###
"""
2a) Fill in code in the template below to implement the function `mini_gamut()` that returns an array of the representable positive numbers for e=0 (with base beta=2).
"""
def mini_gamut():
    """
    Compute positive representable values with exponent zero and 4 digits of precision

    Args: None
    Returns: 1d numpy array of representable values with exponent zero
    """
    # INSERT YOUR CODE HERE

    a = [1]
    b = [0, 1]
    d = np.zeros((0, 4))

    for i in a:
        for j in b:
            for k in b:
                for l in b:
                    row = np.array([i, j, k, l])
                    d = np.vstack((d, row))

    e0_array = []

    for i in d:
        e0_array.append(digits_to_num(0, i, 0))
        
    e0_array = np.array(e0_array)

    return e0_array


def positive_gamut(mg, e_min, e_max):
    """
    Compute the gamut of positive values for a binary system with 4-digit precision and exponents in [-4,3]

    Args:
        mg: float numpy array of values with exponent zero
        e_min, e_max: int values of min and max exponent values
    Returns: numpy array of representable positive floats
    
    """
    # INSERT YOUR CODE HERE

    a = [1]
    b = [0, 1]
    d = np.zeros((0, 4))

    for i in a:
        for j in b:
            for k in b:
                for l in b:
                    row = np.array([i, j, k, l])
                    d = np.vstack((d, row))

    e_array = []

    for i in d:
        for e in range(e_min, e_max+1):
            e_array.append(digits_to_num(0, i, e))
                
    e_array = np.concatenate((mg, np.array(e_array)))
    
    return e_array


def gamut(e_min, e_max):
    """
    Compute the sorted gamut of values for a binary system with 4-digit precision and exponents in [-4,3]

    Args:
        e_min, e_max: int values of min and max exponent values
    Returns: numpy array of representable positive floats
    """
    # INSERT YOUR CODE HERE

    a = [1]
    b = [0, 1]
    d = np.zeros((0, 4))

    for i in a:
        for j in b:
            for k in b:
                for l in b:
                    row = np.array([i, j, k, l])
                    d = np.vstack((d, row))

    gamut_array = []

    for s in range(0, 2):
        for i in d:
            for e in range(e_min, e_max+1):
                gamut_array.append(digits_to_num(s, i, e))

    gamut_array = np.sort(gamut_array)

    return gamut_array


def p2(e_min, e_max):	
	mg = mini_gamut()
	g_pos = positive_gamut(mg, e_min, e_max)
	g = gamut(e_min, e_max)
	plt.plot(g, g, '+')

###
# 3) Write a rounding function and compute relevant error values due to rounding.
#	Construct an array `v` of 501 values equally spaced on the interval [min(g), max(g)].
#	Using your implementation of `rounded` to implement `rounded_array` that returns an array of rounded values corresponding to an input array.
#	Write functions `absolute_error`, `relative_error`, and `u` to compute arrays of values for the absolute error, relative error, and unit roundoff corresponding to the values in the input array `v`.
###
"""
Fill in code below to implement the function `rounded()` that computes the gamut member number closest to a numerical input value.
"""

def rounded(x, g):
    """
    Compute the gamut member closest to a numerical input value

    Args:
        x: float value to be rounded
        g: numpy float array of representable values
    Returns: float rounded value
    """
    # INSERT YOUR CODE HERE

    val = np.zeros(len(x))

    for i in range(len(x)):
        index = np.argmin(np.abs(x[i] - g))
        val[i] = g[index]

    return val

def rounded_array(v, g):
    """
    Compute the gamut members closest to an array of numerical input values

    Args:
        v: float numpy array of input values to be rounded
        g: numpy float array of representable values
    Returns: numpy float array of rounded values
    """
    #INSERT YOUR CODE HERE
    
    val = rounded(v, g)

    return val

def absolute_error(v, g):
    """
    Compute array of absolute errors

    Args:
        v: numpy float array of input values
        g: numpy float array of representable values
    Returns: numpy float array of absolute differences between input and closest representable value
    """
    # INSERT YOUR CODE HERE
    
    rounded_val = rounded_array(v, g)
    val = np.abs(v - rounded_val)

    return val

def relative_error(v, g):
	"""
	Compute array of relative errors

	Args:
		v: numpy float array of input values
		g: numpy float array of representable values
	Returns: numpy float array of relative errors (ratio of absolute error to absolute input value)
	"""
	#INSERT YOUR CODE HERE

	val = absolute_error(v, g) / np.abs(v)

	return val

def unit_roundoff():
	#INSERT YOUR CODE HERE

	beta = 2
	p = 4
	u = (1/2)*beta**(1-p)

	return u

def p3(e_min, e_max, N):
	g = gamut(e_min, e_max)
	v = np.linspace(g[0], g[-1], N)
	abs_err = absolute_error(v, g)
	rel_err = relative_error(v, g)
	u = unit_roundoff()
	u_vals = u * np.ones(N)

	fig, ax = plt.subplots(figsize=(12, 6))

	ax.plot(v, abs_err, color='blue', label='Absolute error')
	ax.plot(v, rel_err, color='black', label='Relative error')
	ax.plot(v, u_vals, color='green', label='Unit roundoff')
	plt.legend()
	plt.ylim([0,0.6])
	plt.show()

def main():
	e_min, e_max = -4,3
	N = 501
	p1()
	p2(e_min, e_max)
	p3(e_min, e_max, N)

if __name__ == '__main__':
	main()




