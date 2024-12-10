import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Data
IMU_Data = [13, 15, 14, 19, 22, 12, 5, 22, 44, 64, 10, -61, -91, -145, -221, -286, -352, -443, -528, -582, -596, -571, -505, -409, -316, -256, -223, -205, -183, -170, -169, -174]
VEL_A_Data = [0, 0, 0, 0, 0, -4, -1, 0, 0, 0, -31, -47, -45, -42, -43, -47, -47, -48, -57, -52, -45, -39, -37, -29, -23, -27, -37, -36, -13, -2, 0, 0]
VEL_B_Data = [1, 0, 0, 0, 0, 0, 0, 7, 8, 9, 9, 19, 26, 21, 16, 17, 15, 15, 19, 30, 40, 52, 63, 67, 58, 53, 53, 41, 21, 5, 0, -4]

# Prepare polynomial fits for the entire datasets
x_full = np.arange(len(IMU_Data))
poly_fits = []
data_sets = [IMU_Data, VEL_A_Data, VEL_B_Data]

for data in data_sets:
    poly_coeffs = np.polyfit(x_full, data, 20)  # Fit 20th-degree polynomial
    poly_fits.append(np.poly1d(poly_coeffs))

# Preparing the figure
fig, axs = plt.subplots(3, 1, figsize=(10, 8))
lines_poly = []
titles = ["IMU Data", "Velocity A Data", "Velocity B Data"]

# Initialize subplots
for i, (ax, title) in enumerate(zip(axs, titles)):
    ax.set_xlim(0, len(IMU_Data))
    if i == 0:
        ax.set_ylim(min(IMU_Data) - 50, max(IMU_Data) + 50)
    else:
        ax.set_ylim(-60, 80)  # Zoom in for velocity data
    ax.set_title(title)
    ax.grid(True)
    lines_poly.append(ax.plot([], [], 'r-', label='Poly Fit')[0])
    ax.legend()

# Function to animate the polynomial fit
def animate_poly(frame):
    step_fraction = 0.005  # Fraction of the dataset length per frame for finer steps
    max_x = frame * step_fraction * len(IMU_Data)  # Dynamically extend the range
    max_x = min(max_x, len(IMU_Data))  # Ensure we don't exceed the dataset length
    for i, (line, poly_fit) in enumerate(zip(lines_poly, poly_fits)):
        x_fit = np.linspace(0, max_x, 1000)  # 1000 points for smooth fit
        y_fit = poly_fit(x_fit)
        line.set_data(x_fit, y_fit)

# Update function for animation
def update(frame):
    animate_poly(frame)
    return lines_poly

# Animation
frames = int(1 / 0.005)  # Number of frames based on step fraction
ani = FuncAnimation(fig, update, frames=frames, interval=5, blit=True)

# Display the plot
plt.tight_layout()
plt.show()
