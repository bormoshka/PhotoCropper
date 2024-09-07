import math
import numpy as np
import matplotlib.pyplot as plt


# Non-linear scaling function
def non_linear_scaling(face_proportion, min_scale_factor=0.1, max_scale_factor=4.0):
    scaling_factor = (0.8 / (math.sqrt(face_proportion) + 0.01) / 1.1) - 1.0
    if scaling_factor > max_scale_factor:
        scaling_factor = max_scale_factor
    if scaling_factor < min_scale_factor:
        scaling_factor = min_scale_factor
    return scaling_factor


# Generate data for the chart
face_proportions = np.linspace(0, 1, 100)  # 100 values between 0 and 1
scaling_factors = [non_linear_scaling(fp) for fp in face_proportions]

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(face_proportions, scaling_factors, label="Scaling Factor", color="blue", linewidth=2)

# Add labels and title
plt.xlabel("Face Proportion (Relative to Image Area)")
plt.ylabel("Scaling Factor")
plt.title("Non-linear Dependency of Scaling Factor on Face Proportion")
plt.grid(True)

# Add a legend
plt.legend()

# Show the chart
plt.show()
