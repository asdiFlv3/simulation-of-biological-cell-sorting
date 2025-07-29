import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from collections import Counter
from numba import njit, int64
from matplotlib.animation import PillowWriter

# 8-neighbor directions (including diagonals)
neighbor_offsets = np.array([(-1,0),(1,0),(0,-1),(0,1), (-1,-1),(-1,1),(1,-1),(1,1)], dtype=np.int64)
# For weighting: 1 for nearest, 1/sqrt(2) for diagonals
neighbor_weights = np.array([1, 1, 1, 1, 1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.float64)

# Define parameters
N_internal_cells = 100 # Number of internal cells (dark and light) [cite: 237, 257]
total_spin_ids = N_internal_cells + 1 # Total unique spin IDs: N internal + 1 for medium [cite: 257, 274]
medium_spin_id = 0 # Designate spin ID 0 as the medium cell 

L = 64 # Lattice size
cx, cy = L//2, L//2
radius = L // 2

# Build a boolean mask of sites inside the circle
y_idx, x_idx = np.ogrid[:L, :L]
mask = (x_idx - cx)**2 + (y_idx - cy)**2 <= radius**2

# cell_type array: Maps spin ID to biological type (0=medium, 1=dark, 2=light)
# Type mapping: 0 = Medium, 1 = Dark, 2 = Light (based on J matrix and common convention)
cell_type = np.zeros(total_spin_ids, dtype=int)
cell_type[medium_spin_id] = 0 # Spin 0 is medium type 

# Create a balanced distribution of dark and light cells
# First half of internal spins are dark, second half are light
half_internal = N_internal_cells // 2
cell_type[1:half_internal+1] = 1  # Dark cells
cell_type[half_internal+1:] = 2    # Light cells

# Initialize spin_grid: all sites *outside* the circular mask are assigned the medium_spin_id.
# All sites *inside* the circular mask are assigned random internal cell spin IDs (1 to N_internal_cells).
spin_grid = np.full((L, L), medium_spin_id, dtype=int) # Fill entire grid with medium_spin_id first 

inside_count = mask.sum()
# Generate random spin IDs for the internal cells (from 1 to N_internal_cells)
# `replace=True` allows multiple sites to have the same spin ID, forming larger initial "cells"
random_internal_ids = np.random.choice(np.arange(1, total_spin_ids), size=inside_count, replace=True)
spin_grid[mask] = random_internal_ids


J = np.array([
    [0, 16, 16],  # Medium (0) with Medium, Dark (1), Light (2) [cite: 328]
    [16, 2, 11],  # Dark (1) with Medium, Dark, Light [cite: 328]
    [16, 11, 14]  # Light (2) with Medium, Dark, Light [cite: 328]
])

# Renaming target_area to target_areas_by_type for clarity and correct usage
# The paper states target area for medium (A_M) is negative to suppress its constraint[cite: 275].
target_areas_by_type = np.array([-1, 40, 40]) # -1 for medium, 40 for dark, 40 for light [cite: 275, 329]

# Check if a point is within the circular region
# Removed default values to ensure explicit passing from calls, or rely on global cx, cy, radius directly within the function

def is_within_circle(x, y, center_x, center_y, radius):
    #Check if a point (x, y) is within the circular region.
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    return distance <= radius

areas = np.zeros(total_spin_ids, dtype=np.int32) # Use int32 for performance with Numba

# In compute_areas, fill this array
def compute_areas(spin_grid, total_spin_ids): # Add total_spin_ids as arg
    counts_arr = np.zeros(total_spin_ids, dtype=np.int32)
    flat = spin_grid.ravel()
    for val in flat: # Manual loop to fill array from counts, Numba compatible
        counts_arr[val] += 1
    return counts_arr

# Initial call:
areas = compute_areas(spin_grid, total_spin_ids)

# New Functions for Calculating Bulk and Edge Lengths
def calculate_metrics(spin_grid, mask, medium_spin_id, cell_type, neighbor_offsets, neighbor_weights, L):
    """
    Calculates the aggregate area (bulk) and various edge lengths.

    Args:
        spin_grid (np.ndarray): The current state of the spin grid.
        mask (np.ndarray): Boolean mask indicating the aggregate's circular region.
        medium_spin_id (int): The spin ID designated for the medium.
        cell_type (np.ndarray): Array mapping spin IDs to cell types.
        neighbor_offsets (np.ndarray): Array of neighbor offsets.
        neighbor_weights (np.ndarray): Array of neighbor weights.
        L (int): Grid size.

    Returns:
        tuple: (aggregate_area, total_interfacial_length, dark_light_adhesion_length)
    """
    aggregate_area = 0
    total_interfacial_length = 0
    dark_light_adhesion_length = 0
    
    # Iterate over each site within the lattice
    for x in range(L):
        for y in range(L):
            current_spin = spin_grid[x, y]
            current_type = cell_type[current_spin]

            # Calculate aggregate area
            if current_spin != medium_spin_id and mask[x, y]:
                aggregate_area += 1

            # Check neighbors for interfacial lengths
            for i in range(len(neighbor_offsets)):
                dx, dy = neighbor_offsets[i]
                w = neighbor_weights[i]
                nx, ny = (x + dx) % L, (y + dy) % L
                neighbor_spin = spin_grid[nx, ny]
                neighbor_type = cell_type[neighbor_spin]

                # Count bonds between different spin IDs (total interfacial length)
                # Only count each bond once (e.g., site (x,y) to neighbor (nx,ny) is same as (nx,ny) to (x,y))
                # So, only count if (x,y) is "lower" in some arbitrary order (e.g., raster scan) than (nx,ny)
                if current_spin != neighbor_spin:
                    total_interfacial_length += w

                # Count specific Dark-Light adhesion length
                # Types: 0=Medium, 1=Dark, 2=Light
                if (current_type == 1 and neighbor_type == 2) or \
                   (current_type == 2 and neighbor_type == 1):
                    dark_light_adhesion_length += w
    
    # Divide total_interfacial_length and dark_light_adhesion_length by 2
    # because each bond is counted twice (once from each side)
    return aggregate_area, total_interfacial_length / 2, dark_light_adhesion_length / 2

def plot_metrics_over_time(steps_list, aggregate_areas, total_interfacial_lengths, dark_light_adhesion_lengths):
    """
    Plots the calculated metrics over simulation steps.
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(steps_list, aggregate_areas, label='Aggregate Area (Bulk)', color='blue')
    plt.plot(steps_list, total_interfacial_lengths, label='Total Interfacial Length', color='red')
    plt.plot(steps_list, dark_light_adhesion_lengths, label='Dark-Light Adhesion Length', color='green')
    
    plt.xlabel('Simulation Steps')
    plt.ylabel('Length / Area')
    plt.title('Potts Model Metrics Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('potts_metrics_over_time.png', dpi=300)
    plt.show()

def create_animation_gif(snapshots, cell_type, total_spin_ids, L, cx, cy, radius, steps, filename='cell_sorting_animation.gif'):
    """
    Creates an animated GIF from the snapshots.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create custom colormap
    colors = ['white', 'black', 'lightgray']
    cmap = matplotlib.colors.ListedColormap(colors)
    
    # Initialize the plot
    cell_type_grid = np.zeros_like(snapshots[0])
    for spin_id in range(total_spin_ids):
        cell_type_grid[snapshots[0] == spin_id] = cell_type[spin_id]
    
    masked_grid = np.ma.masked_where(~is_within_circle(np.arange(L)[:, None], np.arange(L)[None, :], cx, cy, radius), cell_type_grid)
    im = ax.imshow(masked_grid, cmap=cmap, interpolation='nearest', vmin=0, vmax=2)
    ax.set_title(f'Step 0')
    ax.axis('off')
    
    def animate(frame):
        # Convert spin IDs to cell types for visualization
        cell_type_grid = np.zeros_like(snapshots[frame])
        for spin_id in range(total_spin_ids):
            cell_type_grid[snapshots[frame] == spin_id] = cell_type[spin_id]
        
        # Create masked array
        masked_grid = np.ma.masked_where(~is_within_circle(np.arange(L)[:, None], np.arange(L)[None, :], cx, cy, radius), cell_type_grid)
        
        # Update the image
        im.set_array(masked_grid)
        
        # Update title
        if frame == len(snapshots) - 1:
            ax.set_title(f'Final Step {steps}')
        else:
            ax.set_title(f'Step {frame * steps // (len(snapshots) - 1)}')
        
        return [im]
    
    # Create animation
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, animate, frames=len(snapshots), interval=200, blit=True)
    
    # Save as GIF
    writer = PillowWriter(fps=5)
    anim.save(filename, writer=writer)
    plt.close()
    
    print(f"Animation saved as {filename}")


# Calculate the change in Hamiltonian (energy) if a site at (x, y) is changed from its current label to a neighbor's label
# Now includes 8-neighbor interactions with proper weights
# grid: 2D numpy array of cell labels
# cell_types: array mapping each label to a cell type
# target_area: array of target areas for each type
# J: adhesion energy matrix between types
# lam: area constraint strength
# x, y: coordinates of the site to change
# new_label: the label to change the site to
# size: grid size (assumed square)
# Returns: the total energy change (delta H) for the proposed move

def delta_H(spin_grid, target_areas_by_type, J, lam, x, y, new_label): # Renamed parameter
    old_label = spin_grid[x, y]
    if old_label == new_label:
        return 0

    dH_interface = 0
    old_type = cell_type[old_label]
    new_type = cell_type[new_label]
    
    for i in range(len(neighbor_offsets)):
        dx, dy = neighbor_offsets[i]
        w = neighbor_weights[i]
        nx, ny = (x+dx)%L, (y+dy)%L
        neighbor_label = spin_grid[nx, ny]
        neighbor_type = cell_type[neighbor_label]

        dH_interface += w * (J[new_type, neighbor_type] - J[old_type, neighbor_type])

    dH_area = 0
    # Area penalty for the old cell losing a site
    if target_areas_by_type[old_type] > 0: # Check the target area by type [cite: 275]
        dH_area += lam * ( ( (areas[old_label] - 1) - target_areas_by_type[old_type] )**2 - ( areas[old_label] - target_areas_by_type[old_type] )**2 )

    # Area penalty for the new cell gaining a site
    if target_areas_by_type[new_type] > 0: # Check the target area by type [cite: 275]
        dH_area += lam * ( ( (areas[new_label] + 1) - target_areas_by_type[new_type] )**2 - ( areas[new_label] - target_areas_by_type[new_type] )**2 )

    return dH_interface + dH_area

# Attempt a single Monte Carlo move (site label change)
# grid: 2D numpy array of cell labels
# cell_types: array mapping each label to a cell type
# target_area: array of target areas for each type
# J: adhesion energy matrix
# lam: area constraint strength
# T: temperature
# size: grid size

def maybe_take_snapshot(step, spin_grid, snapshots, snapshot_steps, snapshot_idx):
    # Store snapshot if at or past the right step
    while snapshot_idx < len(snapshot_steps) and (step+1) >= snapshot_steps[snapshot_idx]:
        snapshots.append(spin_grid.copy())
        snapshot_idx += 1
    return snapshots, snapshot_idx

def monte_carlo_step(spin_grid, target_areas_by_type, J, lam, T, center_x, center_y, radius): # Removed default args from signature
    # Pick a random site within the circle
    while True:
        x, y = np.random.randint(0, L-1), np.random.randint(0, L-1)
        if is_within_circle(x, y, center_x, center_y, radius):
            break
    
    old_label = spin_grid[x, y]
    
    # Collect all unique neighboring labels (including the medium if a neighbor is outside the aggregate)
    candidate_new_labels = [] # Use a list to allow random.choice easily
    for i in range(len(neighbor_offsets)):
        dx, dy = neighbor_offsets[i]
        nx, ny = (x+dx)%L, (y+dy)%L
        if not is_within_circle(nx, ny, center_x, center_y, radius):
            # If neighbor is outside, it's the medium cell
            candidate_new_labels.append(medium_spin_id)
        else:
            candidate_new_labels.append(spin_grid[nx, ny])
            
    # Remove the old_label from candidates if it's present, unless it's the only option
    # (i.e., if all neighbors have the same spin as the current site, no flip happens)
    unique_candidates = []
    for label in candidate_new_labels:
        if label != old_label:
            unique_candidates.append(label)
    
    if not unique_candidates: # No valid different labels to flip to
        return

    # Choose a new label randomly from the valid unique candidate labels
    new_label = np.random.choice(np.array(unique_candidates))
    
    # Calculate energy change for the proposed move
    dH = delta_H(spin_grid, target_areas_by_type, J, lam, x, y, new_label)
    
    # Metropolis criterion, adjusting for T=0 as per paper
    if T > 0:
        if dH <= 0 or np.random.random() < np.exp(-dH/T):
            spin_grid[x, y] = new_label
            areas[old_label] -= 1
            areas[new_label] += 1
    else: # T == 0
        if dH < 0:
            spin_grid[x, y] = new_label
            areas[old_label] -= 1
            areas[new_label] += 1
        elif dH == 0:
            if np.random.random() < 0.5: # 0.5 probability for dH = 0 at T=0 
                spin_grid[x, y] = new_label
                areas[old_label] -= 1
                areas[new_label] += 1

# Zero-temperature annealing: perform a number of sweeps at T=0 to heal isolated spins
def zero_temp_anneal(spin_grid, target_areas_by_type, J, lam, sweeps=2, center_x=cx, center_y=cy, radius=radius):
    size = spin_grid.shape[0]
    for _ in range(sweeps * size * size):
        # Pass cx, cy, radius explicitly
        monte_carlo_step(spin_grid, target_areas_by_type, J, lam, T=0.0, center_x=center_x, center_y=center_y, radius=radius)



# Set up and run the simulation, collecting snapshots
# size: grid size (square)
# n_cells: number of unique cell labels
# n_types: number of cell types
# T: temperature
# lam: area constraint strength
# steps: number of Monte Carlo steps
# snapshots: number of snapshots to collect
# J_matrix: optional custom adhesion energy matrix (if None, use default)
# per_type_target_area: 
# Returns: tuple (list of grid snapshots, cell_types array)

steps = 3200000
num_snapshots = 128 # intended number of snapshots
snapshots = [spin_grid.copy()]
snapshot_steps = np.linspace(0, steps, num_snapshots, endpoint=True, dtype=int)[1:]
snapshot_idx = 0
lam = 1
T = 10
# Ensure the last snapshot step is exactly the last step
if len(snapshot_steps) > 0:
    snapshot_steps[-1] = steps 

# Perform initial 0K annealing steps
print("Performing initial 0K annealing steps...")
zero_temp_anneal(spin_grid, target_areas_by_type, J, lam, sweeps=2, center_x=cx, center_y=cy, radius=radius)
print("Annealing completed.")

# Recompute areas after annealing, as the annealing steps would have changed cell sizes
areas = compute_areas(spin_grid, total_spin_ids)

# Lists to store metrics for plotting
time_steps = []
aggregate_areas_over_time = []
total_interfacial_lengths_over_time = []
dark_light_adhesion_lengths_over_time = []


for step in range(steps):
    # Pass cx, cy, radius explicitly to monte_carlo_step
    monte_carlo_step(spin_grid, target_areas_by_type, J, lam, T, cx, cy, radius)
    
    # Calculate and store metrics at snapshot intervals
    if step % (steps // num_snapshots) == 0 or step == steps - 1: # Capture metrics at same intervals as snapshots
        current_aggregate_area, current_total_interfacial_length, current_dark_light_adhesion_length = \
            calculate_metrics(spin_grid, mask, medium_spin_id, cell_type, neighbor_offsets, neighbor_weights, L)
        
        time_steps.append(step)
        aggregate_areas_over_time.append(current_aggregate_area)
        total_interfacial_lengths_over_time.append(current_total_interfacial_length)
        dark_light_adhesion_lengths_over_time.append(current_dark_light_adhesion_length)

    snapshots, snapshot_idx = maybe_take_snapshot(step, spin_grid, snapshots, snapshot_steps, snapshot_idx)

# Ensure the final state is captured
if len(snapshots) < num_snapshots:
    snapshots.append(spin_grid.copy())

    # Optional: print progress
    if step % (steps // 10) == 0:
        print(f"Step {step}/{steps} completed.")

print("Simulation completed!")

# Plot the snapshots
fig, axes = plt.subplots(8, 8, figsize=(24, 16))
axes = axes.flatten()  # Flatten to 1D array for easier indexing

# Plot each snapshot
for i, (ax, snap) in enumerate(zip(axes, snapshots)):
    # Convert spin IDs to cell types for visualization
    cell_type_grid = np.zeros_like(snap)
    for spin_id in range(total_spin_ids):
        cell_type_grid[snap == spin_id] = cell_type[spin_id]
    
    # Create a masked array to show only the circular region
    # Pass cx, cy, radius explicitly to is_within_circle
    masked_grid = np.ma.masked_where(~is_within_circle(np.arange(L)[:, None], np.arange(L)[None, :], cx, cy, radius), cell_type_grid)
    
    # Create custom colormap: medium=white, light=gray, dark=black
    # Map cell types: 0=medium (white), 1=dark (black), 2=light (lightgray)
    colors = ['white', 'black', 'lightgray']
    cmap = matplotlib.colors.ListedColormap(colors)
    
    # Plot with custom colormap
    im = ax.imshow(masked_grid, cmap=cmap, interpolation='nearest', vmin=0, vmax=2)
    if i == len(snapshots) - 1:
        ax.set_title(f'Final Step {steps}')
    else:
        ax.set_title(f'Step {i * steps // (len(snapshots) - 1)}')
    ax.axis('off')


plt.suptitle('Potts Model Cell Sorting Progression')
plt.tight_layout()
plt.savefig('cell_sorting_simulation.png', dpi=300, bbox_inches='tight')
plt.show()
print("Simulation completed! Check 'cell_sorting_simulation.png' for the results.")

# Create animated GIF
print("Creating animated GIF...")
create_animation_gif(snapshots, cell_type, total_spin_ids, L, cx, cy, radius, steps, 'cell_sorting_animation.gif')

# Plot metrics over time
plot_metrics_over_time(time_steps, aggregate_areas_over_time, 
                       total_interfacial_lengths_over_time, 
                       dark_light_adhesion_lengths_over_time)

print("Simulation completed! Check 'cell_sorting_simulation.png' for the visual results, 'potts_metrics_over_time.png' for the metrics plots, and 'cell_sorting_animation.gif' for the animation.")
