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
lines_data = []
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
    lines_data.append(ax.plot([], [], 'b-', label='Data')[0])
    lines_poly.append(ax.plot([], [], 'r-', label='Poly Fit')[0])
    ax.legend()

# Function to animate data points
def animate_data(frame):
    x_data = np.arange(frame + 1)
    for i, (line, data) in enumerate(zip(lines_data, data_sets)):
        y_data = data[:frame + 1]
        line.set_data(x_data, y_data)

# Function to animate the polynomial fit
def animate_poly(frame):
    for i, (line, poly_fit) in enumerate(zip(lines_poly, poly_fits)):
        x_fit = np.linspace(0, frame, 500)  # 500 points for smooth fit
        y_fit = poly_fit(x_fit)
        line.set_data(x_fit, y_fit)

# Update function for animation
def update(frame):
    animate_data(frame)
    #animate_poly(frame)
    return lines_data + lines_poly

# Animation
ani = FuncAnimation(fig, update, frames=len(IMU_Data), interval=200, blit=True)

# Display the plot
plt.tight_layout()
plt.show()
