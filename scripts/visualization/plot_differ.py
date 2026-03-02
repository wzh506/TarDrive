import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from matplotlib.collections import LineCollection
import os

# ==========================================
# 1. Data Generation & Helper Functions
# ==========================================

# Set random seed for reproducibility
np.random.seed(42)

# Generate source data: Complex "Two Moons" distribution
n_samples = 1000
X_source, _ = make_moons(n_samples=n_samples, noise=0.05)
# Center and scale the data for better plotting
X_source = (X_source - X_source.mean(axis=0)) / X_source.std(axis=0) * 1.5

# Generate NF target data: Standard Gaussian distribution
X_target_nf = np.random.randn(n_samples, 2) * 0.8

# Create a background grid for visualizing transformations
def create_grid(x_range, y_range, n_lines=15):
    x = np.linspace(x_range[0], x_range[1], n_lines)
    y = np.linspace(y_range[0], y_range[1], n_lines)
    xv, yv = np.meshgrid(x, y)
    
    # Create collections of horizontal and vertical lines
    lines = []
    for i in range(n_lines):
        lines.append(np.stack([xv[i, :], yv[i, :]], axis=1)) # Horizontal lines
        lines.append(np.stack([xv[:, i], yv[:, i]], axis=1)) # Vertical lines
    return lines

# Create initial grid
initial_grid = create_grid((-3, 3), (-3, 3))

# Define a plotting helper function
def plot_flow_comparison(ax, title, source_data, transformed_data, transformed_grid_lines, grid_color, target_is_gaussian=False):
    # Set background color
    ax.set_facecolor('#f0f0f0')
    
    # 1. Plot transformed grid
    lc = LineCollection(transformed_grid_lines, colors=grid_color, linewidths=0.8, alpha=0.5, zorder=1)
    ax.add_collection(lc)
    
    # 2. Plot source data (translucent gray, representing past state)
    ax.scatter(source_data[:, 0], source_data[:, 1], c='gray', alpha=0.2, s=10, label='Source Distribution (Complex)')
    
    # 3. Plot transformed data (colored, representing present state)
    # Use cool color if target is Gaussian, warm color if not
    final_color = 'tab:blue' if target_is_gaussian else 'tab:red'
    label_text = 'Transformed (Normalized to Gaussian)' if target_is_gaussian else 'Transformed (Not Normalized/Folded)'
    ax.scatter(transformed_data[:, 0], transformed_data[:, 1], c=final_color, alpha=0.7, s=15, label=label_text)

    # Decorations
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

# ==========================================
# 2. Define Transformation Functions (Simulating Flows)
# ==========================================

# --- Simulate Normalizing Flow (NF) ---
# This is a smooth, invertible transformation. We simulate this by warping the grid smoothly.
# Note: For visual clarity, data points use generated Gaussian data,
# while the grid uses an analytical invertible function to simulate the visual effect.
def nf_transform_grid(grid_lines):
    transformed = []
    for line in grid_lines:
        x, y = line[:, 0], line[:, 1]
        # Apply a smooth rotation + squeeze transformation (simulating an invertible operation)
        angle = np.sqrt(x**2 + y**2) * 0.5
        x_new = x * np.cos(angle) - y * np.sin(angle)
        y_new = x * np.sin(angle) + y * np.cos(angle)
        transformed.append(np.stack([x_new, y_new*0.8], axis=1))
    return transformed

grid_nf = nf_transform_grid(initial_grid)

# --- Simulate General Flow ---
# This is a non-invertible transformation containing a "folding" operation.
# Using absolute value or other non-monotonic functions causes folding.
def general_transform(data):
    x, y = data[:, 0], data[:, 1]
    # Fold space along the Y-axis: fold the bottom half upwards
    y_new = np.abs(y) - 0.5 
    # Add some non-linear distortion
    x_new = x + np.sin(y_new * 3) * 0.3
    return np.stack([x_new, y_new], axis=1)

def general_transform_grid(grid_lines):
    transformed = []
    for line in grid_lines:
        transformed.append(general_transform(line))
    return transformed

X_target_general = general_transform(X_source)
grid_general = general_transform_grid(initial_grid)


# ==========================================
# 3. Start Plotting
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Left Plot: Normalizing Flow
plot_flow_comparison(
    axes[0], 
    title="Normalizing Flow\n(Clear Goal, Strict Constraints)",
    source_data=X_source,
    transformed_data=X_target_nf, # Target is perfect Gaussian
    transformed_grid_lines=grid_nf,
    grid_color='tab:blue',
    target_is_gaussian=True
)
# Add annotation arrow
axes[0].annotate('Grid Smoothly Warped\nNo Folding (Invertible)', xy=(-2, -2), xytext=(-3, -3.2),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=9, color='tab:blue')

# Right Plot: General Flow
plot_flow_comparison(
    axes[1], 
    title="General Flow\n(No Specific Goal, Potentially Non-invertible)",
    source_data=X_source,
    transformed_data=X_target_general, # Target remains complex and folded
    transformed_grid_lines=grid_general,
    grid_color='tab:red',
    target_is_gaussian=False
)
# Add annotation arrow pointing to folding region
axes[1].annotate('Grid Crossing/Folding\nInformation Lost (Non-invertible)', xy=(0, -0.5), xytext=(-1.5, -3),
                 arrowprops=dict(facecolor='red', arrowstyle='->'), fontsize=9, color='tab:red')

plt.tight_layout()

# Save the image to the current directory
output_filename = "scripts/fig/flow_comparison_en.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Image saved as: {os.path.abspath(output_filename)}")

# Close the figure window to free memory
plt.close(fig)