'''
Homework 4 - ME 535 Winter 2024
__This is an individual assignment, so the answers you submit must be your own work.__
Before you upload the file to Canvas, execute the final version of the file to make sure that it runs properly and produces your intended answers.
Insert your code to implement the functions as specified (without changing function names and/or arguments).
Do not import from libraries other than those that are already imported into the template (such as numpy).
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
TMAX = 100


'''
The models used by engineers often involve linear differential equations. 
A familiar example is the damped linear (or harmonic) oscillator: 
$$\frac{d^2 y}{dt^2} + c \frac{dy}{dt} + y = f(t)$$ or y'' + c*y' + y = f(t)
Linear equations are "friendly" because we can solve them analytically and they support superposition. 
However, the "real world" is not that friendly; most systems are actually nonlinear 
and we typically cannot write down analytic solutions. 
Instead, we often employ numerical methods to compute approximate solutions for more realistic systems 
modeled by nonlinear differential equations. 
The following problems deal with a classic nonlinear second-order ODE, the van der Pol equation, 
which describes an oscillator with nonlinear damping:
$$\frac{d^2 y}{dt^2} -\epsilon(1-y^2) \frac{dy}{dt} + y = 0$$ or y'' - eps*(1-y**2)*y' + y = 0
'''

###
# 1) The Ch. 7 notebook includes an implementation of an Runge-Kutta ODE solver `rk_solve()` 
# that calls a function `rk2_step()` that computes a single step for the second order RK method. 
# (Those codes are included below for your convenience.)
# Below is an embellished version of that code that includes an extra argument so we can pass 
# a value for the parameter `eps` to the rate equation.
# Your initial task is to implement the function `rk4_step` to compute a single step 
# of the 4th order Runge-Kutta method.
# Use `rk_solve` to call your `rk4_step` function to compute a numerical solution for y(t)
# on the interval t=[0,150] with stepsize h=0.1, parameter value eps=0.1 
# and initial conditions y(0)=0.1, y'[0]=0.25
###

vdp_rate = lambda y,t,eps: np.array([y[1], eps*(1-y[0]**2)*y[1] - y[0]])

def euler_step(f,y,t0,t1, param=0):
    """
    compute next value for Euler's method ODE solver
    
    Args:
        f: name of right-hand side function that gives rate of change of y
        y: float value of dependent variable
        t0: float initial value of independent variable
        t1: float final value of independent variable
        param: float parameter in rate function
        
    Returns:
        y_new: float estimated value of y(t1)
    """
    f0 = f(y,t0,param)
    h = t1-t0
    y_new = y + h*f0
    return y_new

def rk2_step(f,y,t0,t1,param):
    """
    compute next value for Euler's method ODE solver
    
    Args:
        f: name of right-hand side function that gives rate of change of y
        y: float value of dependent variable
        t0: float initial value of independent variable
        t1: float final value of independent variable
        param: float parameter in rate function
        
    Returns:
        y_new: float estimated value of y(t1)
    """
    f0 = f(y,t0,param)
    h  = t1-t0
    #compute euler estimate for half step
    y1 = y + 0.5*h*f0
    t1 = t0 + 0.5*h
    #compute midstep rate estimate
    f1 = f(y1,t1,param)
    #take full step using midstep rate estimate 
    y_new = y + h*f1
    return y_new

def rk4_step(f,y,t0,t1,param):
    """
    compute next value for 4th-order Runge-Kutta method ODE solver
    
    Args:
        f: name of right-hand side function that gives rate of change of y
        y: float value of dependent variable
        t0: float initial value of independent variable
        t1: float final value of independent variable
        param: float parameter in the rate function
        
    Returns:
        y_new: float estimated value of y(t1)
    """
    ### INSERT CODE HERE
    f0 = f(y, t0, param)
    h  = t1 - t0
    f1 = f(y + h/2*f0, t0 + h/2, param)
    f2 = f(y + h/2*f1, t0 + h/2, param)
    f3 = f(y + h*f2, t0 + h, param)
    y_new = y + h/6*(f0 + 2*f1 + 2*f2 + f3)
    return y_new

def rk_solve(f,y0,t,param=0, order=4):
    """
    Runge-Kutta solver for systems of 1st order ODEs
    
    Args:
        f: name of right-hand side function that gives rate of change of y
        y0: numpy array of initial float values of dependent variable
        t: numpy array of float values of independent variable
        param: float parameter in rate function
        order: int order of RK solver with allowed values [1,2,4]
        
    Returns:
        y: 2D numpy array of float values of dependent variable
    """
    step_method = rk2_step # Temporary default value until rk4_step is implemented
    #####################
    # Specify the order #
    if 1 == order:
        step_method =  euler_step
    elif 2 == order:
        step_method = rk2_step
    elif 4 == order:
        step_method = rk4_step
    ######################
    # End of order spec. #
    # Start the solver   #
    ######################
    n = t.size
    m = y0.size #determine size of the dependent variable array
    y = [y0] #list to store 1D numpy array for each time
    for i in range(n-1):
        y_new = step_method(f,y[i],t[i],t[i+1],param) #compute next step as before
        y.append(y_new)
    return np.array(y)

def compute_solution(f, eps, steps):
    '''
    Set up time intervals and initial conditions and compute numerical solution
    Args:
        f: function describing rates of change
        eps: float parameter in rate function
    Returns:
        t: 1D numpy float array of independent variable values
        y: 2D numpy float array of dependent variable values
    '''
    t_final = 150.0
    t = np.linspace(0,TMAX,steps+1)
    y0 = np.array([0.1,0.25])
    y = rk_solve(f,y0,t,eps)
    return t,y

# Insert a string describing the steady-state behavior indicated by your simulation.
steady_state1 = 'Displacement becomes stable around $y \\approx 2$ at $t = 60s $'

def plot_solution(t, y, problem, title):
    plt.title("#{}: Numerical solution".format(problem))
    plt.xlim(0,TMAX)
    plt.ylim(-2.5,2.5)
    plt.xlabel("Time t")
    plt.ylabel("Displacement y")
    plt.plot(t,y,'+-')
    plt.grid()
    plt.title(title)
    plt.show()
    

def p1():
    eps = 0.1
    steps = 500
    tmin, tmax = 0., TMAX
    xmin,xmax = -2.5,2.5
    t,y = compute_solution(vdp_rate, eps, steps+1)
    plot_solution(t,y.T[0], 1, steady_state1)
    return

###
# 2) Repeat the numerical solution computation with eps = 10.0.
# 2a) What happens when you try this simulation with 500 steps?
# 2b) What is the steady-state behavior with 5000 steps?
###

# Insert a string describing the steady-state behavior indicated by your simulations.
steady_state2a = 'Not enough steps to capture the oscillations to observe the steady state behavior'

steady_state2b = 'Enough steps to observe the van der pol oscillations as steady state behavior'

def p2():
    eps = 10.
    steps = 500 # Try this value. Describe what happens in reply4. Then change value to 5000
    tmin, tmax = 0., TMAX
    xmin,xmax = -2.5,2.5
    t,y = compute_solution(vdp_rate, eps, steps+1)
    plot_solution(t,y.T[0], 2, steady_state2a)
    steps = 1500
    t,y = compute_solution(vdp_rate, eps, steps+1)
    plot_solution(t,y.T[0],2, steady_state2b)
    return

###
# 3) 
# Compute a  numerical solution for the system of problem 4 using the RKF45 method 
# in the library function scipy.integrate.solve_ivp that incorporates stepsize control.
# Follow the example for simulating the Lotka-Volterra system on the documentation page:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
# You may find it convenient to write a new definition for the van der Pol rates following the LV example.
###


def vdp(t,y,eps):
    ### INSERT CODE HERE
    return np.array([y[1], eps*(1-y[0]**2)*y[1] - y[0]])

def simulate_RKF45():
    '''
    Function to set up proper arguments to perform simulation using ivp_solve.
    See scipy documentation for details.
    Note that ivp_solve may store solution data transposed from what we coded up previously.

    Return:
        t: 1D numpy array of independent variable values
        y: 2D numpy array of dependent variable values
    '''
    ### INSERT CODE HERE
    t_span = [0, TMAX]
    y0 = np.array([0.1,0.25])
    sol = solve_ivp(vdp, t_span, y0, method='RK45', args=[10.])

    return sol.t, sol.y

def p3():
    eps = 10.
    tmin, tmax = 0., TMAX
    xmin,xmax = -2.5,2.5
    t,y = simulate_RKF45()
    step_count = t.shape[0]
    title = f'RK45 takes {step_count} steps.'
    plot_solution(t,y[0],3,title)
    print('RK4 as a fixed step size Runge-Kutta method has less accuracy than RK45 as a variable step size method.')
    return

def simulate_BDF():
    '''
    Function to set up proper arguments to perform simulation using ivp_solve.
    See scipy documentation for details.
    Note that ivp_solve may store solution data transposed from what we coded up previously.

    Return:
        t: 1D numpy array of independent variable values
        y: 2D numpy array of dependent variable values
    '''
    ### INSERT CODE HERE
    t_span = [0, TMAX]
    y0 = np.array([0.1,0.25])
    sol = solve_ivp(vdp, t_span, y0, method='BDF', args=[10.])

    return sol.t, sol.y

def p4():
    eps = 10.
    steps = 500 # Try this value.
    y = np.array([0.1,0.25])
    t = np.linspace(0, TMAX, steps+1)
    t0 = t[0]
    t1 = t[1]
    f0 = vdp(t0,y,eps)
    h  = t1-t0
    print(h)
    #compute euler estimate for half step
    y1 = y + 0.5*h*f0
    t1 = t0 + 0.5*h
    #compute midstep rate estimate
    f1 = vdp(t1,y1,eps)
    #take full step using midstep rate estimate 
    y_new = y + h*f1
    tmin, tmax = 0., TMAX
    xmin,xmax = -2.5,2.5
    t,y = simulate_BDF()
    step_count = t.shape[0]
    title = f'BDF takes {step_count} steps.'
    plot_solution(t,y[0],4,title)
    return

method_comparison = 'The simulation of BDF method seems to be really closed to RK45 method'

def p5():
    print('#5:' + method_comparison)

if __name__ == "__main__":
    p1()
    p2()
    p3()
    p4()
    p5()