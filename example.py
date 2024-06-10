import numpy as np 
import matplotlib.pyplot as plt

def system_dynamics(state, t):
    # Example dynamics: dx/dt = -y, dy/dt = x
    x, y = state
    dxdt = -y
    dydt = x
    return np.array([dxdt, dydt])

def simulate_trajectory(initial_state, t):
    trajectory = [initial_state]
    state = initial_state
    dt = t[1] - t[0]
    
    for time in t[1:]:
        state = state + system_dynamics(state, time) * dt
        trajectory.append(state)
    
    return np.array(trajectory)

# Time points
t = np.linspace(0, 10, 1000)

# Different initial conditions
initial_conditions = [
    np.array([1, 0]),
    np.array([0, 1]),
    np.array([1, 1]),
    np.array([-1, 1])
]

# Plotting trajectories
plt.figure(figsize=(10, 8))
for initial_state in initial_conditions:
    trajectory = simulate_trajectory(initial_state, t)
    plt.plot(trajectory[:, 0], trajectory[:, 1], label=f'Initial: {initial_state}')

# Adding vector field
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)
U = -Y
V = X
plt.quiver(X, Y, U, V, color='gray', alpha=0.5)

plt.title('Trajectories from Different Initial Conditions with Vector Field')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()