
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
from sklearn.metrics import r2_score
from scipy.integrate import solve_ivp

def draw_state_diagrams(true_thetas, estimated_thetas, time_points, plot_file_name):
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(15, 7))

    ax.plot(time_points, true_thetas, '-.', lw=2, label="True pendulum", color='blue')
    ax.plot(time_points, estimated_thetas, '--', lw=1, label="Estimated pendulum", color='red')
    plt.title("Pendulum angle over time")
    plt.xlabel("Time")
    plt.ylabel("Theta")
    plt.legend()

    plt.savefig(plot_file_name, dpi=1300)
    plt.close()


def loop_around_angles(theta: np.ndarray):
    return (np.abs(theta) % (2 * np.pi)) * np.sign(theta)

def finite_difference(values, dt):
    """Calculates derivative as finite difference (This is only approximation). Also assumes equal intervals dt"""
    return [(values[i] - values[i - 1]) / dt for i in range(1, len(values))]

def animate_pendulum_versus(data1, data2, filename='single_pendulum_versus.mp4', interval=50, fps=20):
    x1, y1 = data1
    x2, y2 = data2

    n_frames = len(x1)

    fig, ax = plt.subplots()
    all_x = [x1, x2]
    all_y = [y1, y2]
    ax.set_xlim(min(map(np.min, all_x)) - 1, max(map(np.max, all_x)) + 1)
    ax.set_ylim(min(map(np.min, all_y)) - 1, max(map(np.max, all_y)) + 1)
    ax.set_aspect('equal')
    ax.set_title('Sinlge Pendulums Side by Side')

    line1, = ax.plot([], [], 'x-', lw=2, label="True pendulum", color='blue')
    line2, = ax.plot([], [], 'o-', lw=2, label="Estimated pendulum", color='red')
    ax.legend()

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2

    def update(i):
        # Pendulum 1
        x_vals1 = [0, x1[i]]
        y_vals1 = [0, y1[i]]
        line1.set_data(x_vals1, y_vals1)

        # Pendulum 2
        x_vals2 = [0, x2[i]]
        y_vals2 = [0, y2[i]]
        line2.set_data(x_vals2, y_vals2)

        return line1, line2

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  init_func=init, blit=True, interval=interval)

    writer = animation.FFMpegWriter(fps=fps)
    ani.save(filename, writer=writer)
    plt.close(fig)


def pendulum_motion(t, input, G, L):
    theta, omega = input
    d_theta = omega
    d_omega = (-G/L) * np.sin(theta)

    d_f = np.array([d_theta, d_omega])

    return d_f

def compute_joints_position(thetas: np.ndarray, L):
    x_1,y_1 = np.array([[L * np.sin(theta), -L * np.cos(theta)] for theta in thetas]).T

    return np.array([x_1, y_1])

def compute_thetas_over_time(duration, dt, initial_state, rk_integrator_args, G, L) -> tuple:

    time = np.arange(0, duration, dt)
    t_span = (time[0], time[-1])
    x_train = solve_ivp(pendulum_motion, t_span, initial_state, t_eval=time, args=(G, L), **rk_integrator_args).y
    x_train_theta = x_train[0]
    joints_over_time = compute_joints_position(x_train_theta, L)

    return x_train_theta, joints_over_time, x_train

def plot_progressive_erros(real_theta, estimated_theta, time_points, file_name="Error"):
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(15, 7))

    error = np.power(real_theta-estimated_theta,2)
    mean_error = np.mean(error)
    median_error = np.median(error)
    r2 = r2_score(real_theta, estimated_theta)

    ax.plot(time_points, error, '--', lw=2, label="Error", color='blue')
    ax.plot(time_points, [mean_error for i in range(len(error))],'-', lw=2, label="Mean error", color='green')
    ax.plot(time_points, [median_error for i in range(len(error))],'-', lw=2, label="Median error", color='red')
    ax.text(0.05, 0.95, f"$R^2$ score = {r2:.4f}", transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    plt.title("Squared error between real and estimated angle (rad)")
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.legend()

    plt.savefig(file_name, dpi=1300)
    plt.close()

def animate_pendulum_versus(data1, data2, filename='single_pendulum_versus.mp4', interval=50, fps=20):
    x1, y1 = data1
    x2, y2 = data2

    n_frames = len(x1)

    fig, ax = plt.subplots()
    all_x = [x1, x2]
    all_y = [y1, y2]
    ax.set_xlim(min(map(np.min, all_x)) - 1, max(map(np.max, all_x)) + 1)
    ax.set_ylim(min(map(np.min, all_y)) - 1, max(map(np.max, all_y)) + 1)
    ax.set_aspect('equal')
    ax.set_title('Sinlge Pendulums Side by Side')

    line1, = ax.plot([], [], 'x-', lw=2, label="True pendulum", color='blue')
    line2, = ax.plot([], [], 'o-', lw=2, label="Estimated pendulum", color='red')
    ax.legend()

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2

    def update(i):
        # Pendulum 1
        x_vals1 = [0, x1[i]]
        y_vals1 = [0, y1[i]]
        line1.set_data(x_vals1, y_vals1)

        # Pendulum 2
        x_vals2 = [0, x2[i]]
        y_vals2 = [0, y2[i]]
        line2.set_data(x_vals2, y_vals2)

        return line1, line2

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  init_func=init, blit=True, interval=interval)

    writer = animation.FFMpegWriter(fps=fps)
    ani.save(filename, writer=writer)
    plt.close(fig)
