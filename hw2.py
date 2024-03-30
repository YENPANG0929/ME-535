# HW 2 Template - ME535 W 2023

import numpy as np
from numpy.testing import assert_allclose

###
# 1)
###
'''
This problem involves writing and using an alternate implementation of elemetary row operations. The implementation in the Ch. 2 notebook uses a `for` loop to individually update each entry in the row being altered. Here you will write an alternate version supported by `numpy` that provides a way to refer to a portion of an array including multiple elements including an entire (or partial) row or column.

The related keywords in `numpy include "views", "slice", and "copy". Please read the materials at the following link that provides a nice description of the details:
https://jakevdp.github.io/PythonDataScienceHandbook/02.02-the-basics-of-numpy-arrays.html
'''

###
# 1a)
###
'''
Write a new version of `row_op` that operates row-wise instead of element-wise.
'''
def row_op(A,c,i0,i1):
    """
    perform elementary row operations
    if i0==i1, multiply row i0 by constant c
    if i0!=i1, add c*(row i0) to row i1
    
    Args:
        A: 2D numpy array representing a matrix
        c: multiplicative constant
        i0,i1: row indices
    """
	# INSERT CODE HERE

    # m, n = A.shape
    # if  i0<0 or i1<0 or i0>=m or i1>=m:
    #     print("WARNING: Invalid index specifications. Each index value i must satisfy 0<=i<#rows.")
    
    m,n = A.shape

    if  i1<0 or i1<0 or i0>=m or i1>=m:
        print("WARNING: Invalid index specifications. Each index value i must satisfy 0<=i<#rows.")

    if i0 == i1:
        A[i0] *= c
    else:
        A[i1] += c*A[i0]

    return A

###
# 1a)
###
'''
Previously we skipped one of the elementary row operations; i.e. swapping rows. Now is the time to fill in that gap. Write code for a row-wise implementation of `swap_rows(A, i0, i1)` that exchanges rows `i0` and `i1` in the array `A`.
'''
def swap_rows(A,i0,i1):
    """
    perform elementary row operation to swap rows i0 and i1
    
    Args:
        A: 2D numpy array representing a matrix
        i0,i1: integer row indices
    """
    # INSERT CODE HERE

    m, n = A.shape
    if  i0<0 or i1<0 or i0>=m or i1>=m:
        print("WARNING: Invalid index specifications. Each index value i must satisfy 0<=i<#rows.")

    A[[i0, i1]] = A[[i1, i0]]

    return A

def p1():
	print("Problem 1:")
	mat = np.eye(3)
	row_op(mat,2,1,1)
	ans = np.array([[1,0,0],[0,2,0],[0,0,1]], dtype = np.float64)
	assert_allclose(mat, ans, err_msg='failed first row op test')

	mat = np.eye(3)
	row_op(mat,3,1,2)
	ans = np.array([[1,0,0],[0,1,0],[0,3,1]], dtype = np.float64)
	assert_allclose(mat, ans, err_msg='failed second row op test')

	mat = np.eye(3)
	swap_rows(mat,0,1)
	ans = np.array([[0,1,0],[1,0,0],[0,0,1]], dtype = np.float64)
	assert_allclose(mat, ans, err_msg='failed row swap test')




###
#2)
###
'''
2a) Write code to implement the matrix iteation scheme for computing the dominant eigenvalue/eigenvector (associated with the eigenvalue with largest magnitude) of a square matrix.
Here is one verion of a simple pseudo-code (that needs termination conditions)
Choose an initial vector u0 such that ∥u0∥ = 1
for k = 1, 2, . . . do
	v^{(k)} = A u^{(k−1)}
	u^{(k)} = v^{(k)}/∥v^{(k)}∥
end
'''
def dominant_eigen_iteration(A, u0, tol, max_iters):
    """
    compute dominant eigenvctor and eigenvalue for square matrix
    
    Args:
        A:  nxn numpy array representing a matrix
        u0: initial estimate of eigenvector (1D numpy array)
        tol: float relative error termination criterion
        max_iters: integer iteration count termination criterion
    Returns:
        lambda: float dominant eigenvalue
		v: dominant eigenvector 1d float numpy array 
    """
	# INSERT CODE HERE

    u = 0

    for k in range(max_iters+1):
        v = np.dot(A, u0)
        l = np.dot(u0, v)
        u0 = v / np.linalg.norm(v)
        
        if abs(l - u) / abs(l) < tol:
            break

        u = l

    return l, v

'''
2b) Write code to implement the "inverse matrix iteration" scheme for computing the "recessive" eigenvalue/eigenvector pair (associated with the eigenvalue with smallest magnitude) of a square matrix. 
Here is a simple pseudo-code (again missing termination conditions)

Choose an initial vector u0 such that ∥u0∥ = 1
for k = 1, 2, . . . do
	Solve A v^{(k)} = u^{(k−1)}
	u^{(k)} = v^{(k)}/∥v^{(k)}∥
end

Use `numply.linalg.solve` in your implementation.
'''
def recessive_eigen_iteration(A, u0, tol, max_iters):
    """
    compute dominant eigenvctor and eigenvalue for square matrix
    
    Args:
        A:  nxn numpy array representing a matrix
        u0: initial estimate of eigenvector (1D numpy array)
        tol: float relative error termination criterion
        max_iters: integer iteration count termination criterion
    Returns:
        lambda: float dominant eigenvalue
		v: dominant eigenvector 1d float numpy array 
    """
	# INSERT CODE HERE

    u = 0

    for k in range(max_iters+1):
        v = np.dot(np.linalg.inv(A), u0)
        l = np.dot(u0, v)
        u0 = v / np.linalg.norm(v)
        
        if abs(l - u) / abs(l) < tol:
            break

        u = l

    return l, v

'''
2c) Use the functions you implemented in parts a and b to create a function that computes an estimate of the condition number of a square matrix.
'''
def condition(A, u0, tol, max_iters):
	'''
	Compute numerical estimate of condition number of a matrix based on eigenspectrum
	Args:
		A: 2D numpy array representing the matrix
		u0: 1D numpy array that serves as initial guess for eignvector iteration
		tol: float residual for termination
		max_iters: int bound on number of iterations
	Returns: float estimate of condition number
	'''
	# INSERT CODE HERE
	
	max_lambda = dominant_eigen_iteration(A, u0, tol, max_iters)[0]
	min_lambda = recessive_eigen_iteration(A, u0, tol, max_iters)[0]
	condition_num = np.abs(max_lambda / min_lambda)
    
	return condition_num

###
# 3) This problem involves functions to support QR factorization.
###
'''
3a) Write code to implement a function to compute the component of vector v along the direction of vector u.
'''
def component_of_along(v, u):
	'''
	Compute component of vector v along direction of vector u
	Args:
		v,u: 1d numpy arrays
	Returns: 1d numpy array representing the component of v along u
	'''
	# INSERT CODE HERE
	
	v_onto_u = np.dot(v, u) / np.dot(u, u) * u

	return v_onto_u

'''
3b) In class, we discussed QR factorization based on Gram Scmidt orthogonalization. In that approach, an orthogonal basis is constructed by subtracting from each new candidate basis vector the components along the direction of each already-computed entry in the (numerically) orthogonal set. That approach can run into precision issues (because candidate basis vectors that lie close to the space spanned by the existing basis vectors can lead to catastrophic cancellation when components are subtracted). That issue led to the creation of other approaches. The one that is the focus of this problem involves Householder reflections. The basic idea is to compute the vector that arises when an input vector v is reflected about the plane normal to a specified vector u.

Fill in code below to implement a function to compute the Householder reflection. This should be pretty straightforward if you usethe function component_of_along that you implemented in 3a.
'''
def reflect(v,u):
	'''
	Compute reflection of vector v across mirror hyperplane with normal vector u
	Args:
		v,u: 1d numpy arrays
	Returns: 1d numpy array representing the refelction of v
	'''
	# INSERT CODE HERE

	Fv = v - 2*component_of_along(v, u)

	return Fv

'''
3c) Householder made good use of this basic reflection operation. First, he noted that the reflection operation corresponds to multiplication of the input vector v by an orthogonal matrix Q_u. Think about why that is true! Then, he figured out how to pick the mirror normal that would produce a reflected vector e0 that lies along the first coordinate axis; i.e. Q_u(v) = norm(v)*e0. Convince yourself that the choice u = c*(v - norm(v)*e0) works for any choice of the constant c. In particular, choose c=1 so u = v - norm(v)*e0

NOTE: The slightly more complicated choice u = v - sign(v[0])*norm(v)*e0 prevents the possibility of catastrophic cancellation errors.

Insert code below to produce the orthogonal matrix that reflects a given input vector to become a multiple of e0.
'''
def reflect_to_e0(v):
	'''
	Compute the matrix that rotates a given vector to the e0 direction
	Args:
		u: 1D numpy array representing vector to rotate
	Returns:
		reflection: 2D numpy array representing the rotation matrix
	'''
	# INSERT CODE HERE
	
	e0 = np.zeros_like(v)
	e0[0] = 1
	v = v - np.sign(v[0])*np.linalg.norm(v)*e0
	Q_u = np.eye(len(v)) - 2*np.outer(v, v)/np.dot(v, v)
	
	return Q_u

'''
3d) Use the function you implemented for part 3c to implement a function that does the first step of QR factorization.
'''
def Householder(A):
	'''
	Compute QR0 partial matrix factorization based on Householder reflection
	Args:
		A: 2D numpy array representing matrix to be factored
	Returns
		Q: 2D float numpy array representing othrogonal factor
		R0: 2D float numpy array representing whose first column is e0
	'''
	# INSERT CODE HERE
	
	v = A[:, 0]
	Q = reflect_to_e0(v)
	R0 = np.dot(Q, A)

	return Q, R0


np.set_printoptions(precision=3)

def p2():
	print("Problem 2:")
	Q = 0.5*np.sqrt(2)*np.array([[1,1,0],[1,-1,0],[0,0,np.sqrt(2)]])
	lam = np.array([[10,0,0],[0,2,0],[0,0,-1]])
	mat = Q.T @ (lam @ Q)
	u0 = np.array([1,1,1])
	tol = 1e-2
	max_iters = 30
	lambda_0, v_0 = dominant_eigen_iteration(mat, u0, tol, max_iters)
	print(f"dominant:  {lambda_0:.2f}, {v_0}")
	# mat = np.array([[3,0,0],[0,2,0],[0,0,-1]])
	u0 = np.array([1,1,1])
	lambda_min, v_min = recessive_eigen_iteration(mat, u0, tol, max_iters)
	print(fr"recessive:  {lambda_min:.2f}, {v_min}")
	kappa = condition(mat,u0, tol, max_iters)
	print("Condition number: ", kappa)

def p3():
	print("Problem 3:")
	u = np.array([1,1,1])
	v = np.array([3,2,1])
	A = np.array([[1,1,1], [1,2,3],[3,2,1]])
	comp = component_of_along(v,u)
	print("component: ", comp)
	print("reflection: ", reflect(v,u))
	Q = reflect_to_e0(v)
	print("refl. matrix: \n", reflect_to_e0(v))
	print("reflected: \n", np.dot(Q,v))
	Q,R = Householder(A)
	print("Householder:") 
	print("Q = \n", Q, "\nR = \n", R)
	res = np.dot(Q,R)-A
	print("Check: residual = \n", res)
	print("Passed test? ", np.allclose(np.dot(Q,R),A) and np.allclose(R[1:,0], np.zeros(2)))


if __name__ == '__main__':
	p1()
	p2()
	p3()