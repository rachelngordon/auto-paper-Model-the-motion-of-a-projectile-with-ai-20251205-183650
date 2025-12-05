# ==== main.py ====
import numpy as np
import matplotlib.pyplot as plt

# Physical constants
g = 9.81  # m/s^2
mass = 1.0  # kg

# Drag coefficients (chosen for illustrative purposes)
k_linear = 0.1   # kg/s for linear drag
k_quad = 0.01    # kg/m for quadratic drag


def acceleration(v, drag_model):
    """Return acceleration vector given velocity vector v and drag model.
    v: np.array([vx, vy])
    drag_model: 'none', 'linear', or 'quadratic'
    """
    ax = 0.0
    ay = -g
    speed = np.linalg.norm(v)
    if drag_model == 'linear':
        drag = -k_linear * v / mass
        ax += drag[0]
        ay += drag[1]
    elif drag_model == 'quadratic':
        if speed != 0:
            drag = -k_quad * speed * v / mass
            ax += drag[0]
            ay += drag[1]
    # 'none' adds no drag
    return np.array([ax, ay])


def simulate(v0, theta_deg, drag_model, dt=0.01, max_time=100.0):
    """Simulate projectile motion.
    Returns arrays of x and y positions.
    """
    theta = np.radians(theta_deg)
    v = np.array([v0 * np.cos(theta), v0 * np.sin(theta)])
    pos = np.array([0.0, 0.0])
    xs = [pos[0]]
    ys = [pos[1]]
    t = 0.0
    while pos[1] >= 0 and t < max_time:
        # RK4 integration step
        a1 = acceleration(v, drag_model)
        k1v = a1 * dt
        k1x = v * dt

        a2 = acceleration(v + 0.5 * k1v, drag_model)
        k2v = a2 * dt
        k2x = (v + 0.5 * k1v) * dt

        a3 = acceleration(v + 0.5 * k2v, drag_model)
        k3v = a3 * dt
        k3x = (v + 0.5 * k2v) * dt

        a4 = acceleration(v + k3v, drag_model)
        k4v = a4 * dt
        k4x = (v + k3v) * dt

        v = v + (k1v + 2*k2v + 2*k3v + k4v) / 6.0
        pos = pos + (k1x + 2*k2x + 2*k3x + k4x) / 6.0
        xs.append(pos[0])
        ys.append(pos[1])
        t += dt
    return np.array(xs), np.array(ys)


def experiment_trajectory_comparison():
    v0 = 50.0  # m/s
    angle = 45.0  # degrees
    models = {
        'No drag': 'none',
        'Linear drag': 'linear',
        'Quadratic drag': 'quadratic'
    }
    plt.figure(figsize=(8, 6))
    for label, model in models.items():
        x, y = simulate(v0, angle, model)
        plt.plot(x, y, label=label)
    plt.title('Projectile trajectories with different drag models')
    plt.xlabel('Horizontal distance (m)')
    plt.ylabel('Vertical height (m)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('trajectory_comparison.png')
    plt.close()


def experiment_range_vs_angle():
    v0 = 50.0  # m/s
    angles = np.arange(10, 85, 5)  # 10° to 80° inclusive
    ranges = []
    for ang in angles:
        x, y = simulate(v0, ang, 'quadratic')
        # Interpolate to find more accurate landing point (y=0)
        if len(y) < 2:
            ranges.append(0.0)
            continue
        if y[-1] > 0:
            # did not land within max_time, treat as last x
            ranges.append(x[-1])
            continue
        # Linear interpolation between last two points where sign changes
        x1, y1 = x[-2], y[-2]
        x2, y2 = x[-1], y[-1]
        if y2 == y1:
            xr = x2
        else:
            xr = x1 - y1 * (x2 - x1) / (y2 - y1)
        ranges.append(xr)
    ranges = np.array(ranges)
    plt.figure(figsize=(8, 6))
    plt.plot(angles, ranges, marker='o')
    plt.title('Range vs Launch Angle (quadratic drag)')
    plt.xlabel('Launch angle (degrees)')
    plt.ylabel('Horizontal range (m)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('range_vs_angle.png')
    plt.close()
    # Determine angle with maximum range
    max_idx = np.argmax(ranges)
    optimal_angle = angles[max_idx]
    optimal_range = ranges[max_idx]
    return optimal_angle, optimal_range


def main():
    # Experiment 1: trajectory comparison
    experiment_trajectory_comparison()
    # Experiment 2: range vs angle
    optimal_angle, optimal_range = experiment_range_vs_angle()
    # Primary answer: optimal launch angle (degrees) for maximum range under quadratic drag
    print('Answer:', optimal_angle)

if __name__ == '__main__':
    main()

