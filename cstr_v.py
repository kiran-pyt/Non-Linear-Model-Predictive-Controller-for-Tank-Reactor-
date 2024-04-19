import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define model equations
def model(t, y, u):
    # Unpack the state variables
    Ca, Cr = y
    
    # Unpack the input variables
    q, Ca0, k01, k02, k03, k04, E_T1, E_T2, E_T3, E_T4, T2 = u
    
    # Rate equations
    k1 = k01 * np.exp(-E_T1 * (1 / T2 - 1))
    k2 = k02 * np.exp(-E_T2 * (1 / T2 - 1))
    k3 = k03 * np.exp(-E_T3 * (1 / T2 - 1))
    k4 = k04 * np.exp(-E_T4 * (1 / T2 - 1))
    
    dCa = q * (Ca0 - Ca) - k1 * Ca + k4 * Cr
    dCr = q * (1 - Ca0 - Cr) + k1 * Ca + k3 * (1 - Ca - Cr) - (k2 + k4) * Cr
    
    return [dCa, dCr]

# Define parameters
E_T = np.array([8.33, 10.0, 50, 83.3])
k0 = np.array([1, 0.7, 0.1, 0.006])

# NMPC parameters
p = 10
m = 7
Nsim = 40
delta_t = 0.1
Qy = np.diag([2.4, 5.67])
Qu = np.diag([25, 25])
umin = np.array([0.75, 0.5])
umax = np.array([0.85, 1.1])  # Adjusted upper bounds
delta_umin = np.array([-0.1, -0.1])
delta_umax = np.array([0.1, 0.1])

# Initial conditions
ynow = np.array([0.68, 0.28])
u_init = np.array([0.8, 0.8])
yref = np.array([[0.324], [0.406]])
big_R = np.tile(yref, (2 * p, 1))  # Replicate yref p times along the first axis
Ca0 = 0.8

def mpc_cost_function(u_delta, ynow, prev_u):
    u = u_delta[:2]
    du = u - prev_u
    nt = p * 4
    ts1 = np.linspace(0, p, nt)
    u1 = [u[0], Ca0, k0[0], k0[1], k0[2], k0[3], E_T[0], E_T[1], E_T[2], E_T[3], u[1]]
    solution = solve_ivp(model, (0, p), ynow, args=(u1,), t_eval=ts1)
    y_pred = solution.y
    error_y = y_pred - big_R.T  # Transpose big_R to match dimensions
    cost = np.sum(np.dot(error_y.T, Qy.dot(error_y)))
    
    # Add penalty on the change in control inputs
    cost += np.sum(np.dot(du.T, Qu.dot(du)))
    
    return cost

# Define constraints
def constraint_func(u_delta):
    u = u_delta[:2]
    delta_u = u_delta[2:]
    constraints = []
    for i in range(len(u)):
        constraints.append(u[i] - umin[i])
        constraints.append(umax[i] - u[i])
        constraints.append(delta_u[i] - delta_umin[i])
        constraints.append(delta_umax[i] - delta_u[i])
    return constraints

# Initialize arrays for saving data
tm = np.linspace(0.0, 3.5, Nsim)
ym = np.zeros((Nsim, 2))
um = np.zeros((Nsim, 2))

# Perform NMPC
prev_u_delta = np.concatenate((u_init, np.zeros(2)))  # Initialize previous control input and delta_u
for i in range(Nsim):
    # Define initial guess for control inputs
    u_guess = np.concatenate((u_init, np.zeros(2)))
    
    # Call the controller to find the predicted optimal control inputs
    res = minimize(mpc_cost_function, u_guess, args=(ynow, prev_u_delta[:2]), method='SLSQP', 
                   constraints={'type': 'ineq', 'fun': constraint_func}, 
                   #bounds=[(umin[0], umax[0]), (umin[1], umax[1]), (delta_umin[0], delta_umax[0]), (delta_umin[1], delta_umax[1])])
                   bounds=[(umin[0], umax[0]), (umin[1], umax[1]), (delta_umin[0], delta_umax[0]), (delta_umin[1], delta_umax[1])])
    u_delta_opt = res.x
    u_opt = u_delta_opt[:2]  # Extract control inputs
    
    u1 = [u_opt[0], Ca0, k0[0], k0[1], k0[2], k0[3], E_T[0], E_T[1], E_T[2], E_T[3], u_opt[1]]
    # Integrate the model forward in time using the optimal control inputs
    t_span = [tm[i], tm[i] + delta_t]
    solution = solve_ivp(model, t_span, ynow, args=(u1,), t_eval=[tm[i] + delta_t])
    
    # Update the current state with the final state of the integration
    ynow = solution.y[:, -1]
    
    # Save current control input and delta_u for next iteration
    prev_u_delta = u_delta_opt
    
    # Save data for plotting or analysis
    ym[i, :] = ynow
    um[i, :] = u_opt

# Plotting ym
plt.figure(figsize=(10, 6))
plt.plot(tm, ym[:, 0], label='Ca')
plt.plot(tm, ym[:, 1], label='Cr')
plt.xlabel('Time')
plt.ylabel('State Variables')
plt.title('Evolution of State Variables')
plt.legend()
plt.grid(True)
plt.show()

# Plotting um
plt.figure(figsize=(10, 6))
plt.plot(tm, um[:, 0], label='Control Input 1')
plt.plot(tm, um[:, 1], label='Control Input 2')
plt.xlabel('Time')
plt.ylabel('Control Inputs')
plt.title('Evolution of Control Inputs')
plt.legend()
plt.grid(True)
plt.show()
