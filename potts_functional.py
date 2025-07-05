import numpy as np
import matplotlib.pyplot as plt
import random

# 8-neighbor directions (including diagonals)
neighbor_offsets = [(-1,0),(1,0),(0,-1),(0,1), (-1,-1),(-1,1),(1,-1),(1,1)]
# For weighting: 1 for nearest, 1/sqrt(2) for diagonals
neighbor_weights = [1, 1, 1, 1, 1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2)]

# Check if a point is within the circular region
def is_within_circle(x, y, center_x=25, center_y=25, radius=25):
    """
    Check if a point (x, y) is within the circular region.
    """
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    return distance <= radius

# Calculate the area (number of sites) belonging to a given cell label
# grid: 2D numpy array of cell labels
# label: the cell label to count
# center_x, center_y, radius: circle parameters
# Returns: integer area of the cell (only within circle)

def area(grid, label, center_x=25, center_y=25, radius=25):
    size = grid.shape[0]
    count = 0
    for x in range(size):
        for y in range(size):
            if grid[x, y] == label and is_within_circle(x, y, center_x, center_y, radius):
                count += 1
    return count

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

def delta_H(grid, cell_types, target_area, J, lam, x, y, new_label, size):
    old_label = grid[x, y]  # Current label at (x, y)
    if old_label == new_label:
        return 0  # No change if labels are the same

    old_type = cell_types[old_label]  # Type of the old cell
    new_type = cell_types[new_label]  # Type of the new cell

    dH_interface = 0  # Change in interface (boundary) energy
    # Loop over 8 neighbors (including diagonals)
    for (dx, dy), w in zip(neighbor_offsets, neighbor_weights):
        nx, ny = (x+dx)%size, (y+dy)%size  # Periodic boundary conditions
        neighbor_label = grid[nx, ny]
        neighbor_type = cell_types[neighbor_label]
        # If neighbor is same as old label, removing this boundary increases energy
        if neighbor_label == old_label:
            dH_interface += w * J[old_type, neighbor_type]
        # If neighbor is same as new label, adding this boundary decreases energy
        if neighbor_label == new_label:
            dH_interface -= w * J[new_type, neighbor_type]

    # Area constraint: penalize deviation from target area, only if target > 0
    area_old = area(grid, old_label)  # Area of old cell
    area_new = area(grid, new_label)  # Area of new cell
    target_old = target_area[old_type]  # Target area for old cell type
    target_new = target_area[new_type]  # Target area for new cell type
    dH_area = 0  # Change in area penalty
    if target_old > 0:
        dH_area += lam * ((area_old-1-target_old)**2 - (area_old-target_old)**2)
    if target_new > 0:
        dH_area += lam * ((area_new+1-target_new)**2 - (area_new-target_new)**2)

    return dH_interface + dH_area  # Total energy change


# Attempt a single Monte Carlo move (site label change)
# grid: 2D numpy array of cell labels
# cell_types: array mapping each label to a cell type
# target_area: array of target areas for each type
# J: adhesion energy matrix
# lam: area constraint strength
# T: temperature
# size: grid size

def maybe_take_snapshot(step, grid, snapshots, snapshot_steps, snapshot_idx):
    # Store snapshot if at or past the right step
    while snapshot_idx < len(snapshot_steps) and (step+1) >= snapshot_steps[snapshot_idx]:
        snapshots.append(grid.copy())
        snapshot_idx += 1
    return snapshots, snapshot_idx

def monte_carlo_step(grid, cell_types, target_area, J, lam, T, size, center_x=25, center_y=25, radius=25):
    # Pick a random site within the circle
    while True:
        x, y = random.randint(0, size-1), random.randint(0, size-1)
        if is_within_circle(x, y, center_x, center_y, radius):
            break
    
    old_label = grid[x, y]
    # Pick a random neighbor direction (from 8 neighbors)
    idx = random.randint(0, 7)
    dx, dy = neighbor_offsets[idx]
    nx, ny = (x+dx)%size, (y+dy)%size
    
    # Only allow moves to neighbors within the circle
    if not is_within_circle(nx, ny, center_x, center_y, radius):
        return  # Reject moves outside the circle
    
    new_label = grid[nx, ny]
    if old_label == new_label:
        return  # Do nothing if same label
    # Calculate energy change for the proposed move
    dH = delta_H(grid, cell_types, target_area, J, lam, x, y, new_label, size)
    # Metropolis criterion: accept if lowers energy, or with probability exp(-dH/T)
    if dH <= 0 or (T > 0 and random.random() < np.exp(-dH/T)):
        grid[x, y] = new_label  # Accept the move

# Zero-temperature annealing: perform a number of sweeps at T=0 to heal isolated spins
def zero_temp_anneal(grid, cell_types, target_area, J, lam, sweeps=2, center_x=25, center_y=25, radius=25):
    size = grid.shape[0]
    for _ in range(sweeps * size * size):
        monte_carlo_step(grid, cell_types, target_area, J, lam, T=0.0, size=size, center_x=center_x, center_y=center_y, radius=radius)

def compute_boundary_lengths(grid, cell_types):
    """
    Returns (bulk_boundary_length, edge_boundary_length) for the current grid.
    - bulk: boundaries between different cell types (not medium)
    - edge: boundaries between cell and medium
    """
    size = grid.shape[0]
    bulk = 0
    edge = 0
    for x in range(size):
        for y in range(size):
            label = grid[x, y]
            t = cell_types[label]
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = (x+dx)%size, (y+dy)%size
                nlabel = grid[nx, ny]
                nt = cell_types[nlabel]
                if t == 0 and nt == 0:
                    continue  # both medium, skip
                if t == 0 or nt == 0:
                    edge += 1
                elif t != nt:
                    bulk += 1
    # Each boundary is counted twice (once from each side), so divide by 2
    return bulk // 2, edge // 2



# Set up and run the simulation, collecting snapshots
# size: grid size (square)
# n_cells: number of unique cell labels
# n_types: number of cell types
# T: temperature
# lam: area constraint strength
# steps: number of Monte Carlo steps
# snapshots: number of snapshots to collect
# J_matrix: optional custom adhesion energy matrix (if None, use default)
# per_type_target_area: optional per-type target area (if None, use default)
# Returns: tuple (list of grid snapshots, cell_types array)

def run_simulation(size, n_cells, n_types, T, lam, steps, snapshots, J_matrix, per_type_target_area, center_x=25, center_y=25, radius=25):
    # Assign each cell label a type (1 or 2) in a balanced way
    cell_types = np.zeros(n_cells+1, dtype=int)
    cell_types[0] = 0  # Medium
    half = n_cells // 2
    cell_types[1:half+1] = 1
    cell_types[half+1:] = 2
    # Set per-type target area; medium unconstrained (target_area[0] < 0)
    if per_type_target_area is not None:
        target_area = np.array(per_type_target_area)
    else:
        # Calculate area within circle for target areas
        circle_area = 0
        for x in range(size):
            for y in range(size):
                if is_within_circle(x, y, center_x, center_y, radius):
                    circle_area += 1
        target_area = np.zeros(n_types+1)
        target_area[1:] = circle_area // n_cells
        target_area[0] = -1  # medium unconstrained
    # Set up adhesion energy matrix J
    if J_matrix is not None:
        J = J_matrix  # Use user-provided matrix
    else:
        J = np.ones((n_types+1, n_types+1)) * 16  # Default high
        np.fill_diagonal(J, 8)  # Lower for same type
        J[0, :] = J[:, 0] = 20  # Medium has highest adhesion
    # Initialize grid with cells only within the circle
    grid = np.zeros((size, size), dtype=int)  # Start with all medium (0)
    
    # Place cells randomly within the circle
    cell_labels = list(range(1, n_cells + 1))
    for x in range(size):
        for y in range(size):
            if is_within_circle(x, y, center_x, center_y, radius):
                grid[x, y] = random.choice(cell_labels)

    # Prepare to store snapshots
    num_snapshots = snapshots  # snapshots is the intended number of snapshots
    snapshots = [grid.copy()]
    snapshot_steps = np.linspace(0, steps, num_snapshots, endpoint=True, dtype=int)[1:]
    snapshot_idx = 0
    # Ensure the last snapshot step is exactly the last step
    if len(snapshot_steps) > 0:
        snapshot_steps[-1] = steps

    for step in range(steps):
        monte_carlo_step(grid, cell_types, target_area, J, lam, T, size, center_x, center_y, radius)
        snapshots, snapshot_idx = maybe_take_snapshot(step, grid, snapshots, snapshot_steps, snapshot_idx)

    # If not enough snapshots (e.g., if steps < num_snapshots), pad with final state
    diff = num_snapshots - len(snapshots)
    for _ in range(diff):
        snapshots.append(grid.copy())

    return snapshots, cell_types, snapshot_steps

'''
def plot_snapshots(snapshots, snapshot_steps=None):
    num_snapshots = len(snapshots)
    fig, axes = plt.subplots(1, num_snapshots, figsize=(3*num_snapshots, 3))
    if num_snapshots == 1:
        axes = [axes]
    
    # Define colors for different cell types
    colors = ['white', 'red', 'blue']  # medium, type1, type2
    
    for i, (ax, snap) in enumerate(zip(axes, snapshots)):
        label = f'Step 0' if i == 0 else (f'Step {snapshot_steps[i-1]}' if snapshot_steps is not None and i > 0 else f'Snapshot {i}')
        
        # Clear the axis
        ax.clear()
        
        # Plot each cell type separately with different colors and sizes
        for cell_type in range(1, len(colors)):  # Skip medium (type 0)
            # Find all positions of this cell type
            positions = np.where(cell_types[snap] == cell_type)
            if len(positions[0]) > 0:
                # Use larger point size for better visibility
                ax.scatter(positions[1], positions[0], 
                          c=colors[cell_type], 
                          s=50,  # Point size
                          marker='o', 
                          alpha=0.8,
                          edgecolors='black',
                          linewidth=0.5)
        
        # Set axis properties for better visualization
        ax.set_xlim(-1, snap.shape[1])
        ax.set_ylim(snap.shape[0], -1)  # Invert y-axis to match grid coordinates
        ax.set_aspect('equal')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
        
    plt.suptitle('Potts Model Cell Sorting Progression (Scatter Plot)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
'''

def plot_snapshots_rounded(snapshots, snapshot_steps=None, point_size=50):
    """
    Alternative scatter plot with larger, more rounded cell representations
    and better visual separation between cell types.
    Only shows points within a circle of radius 25 centered at (25,25).
    """
    num_snapshots = len(snapshots)
    
    # Create 4x2 subplot layout
    fig, axes = plt.subplots(2, 4, figsize=(12, 16))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing
    
    # Define colors for different cell types with better contrast
    colors = ['white', '#FFBE7A', '#8ECFC9']  # medium, type1 (coral), type2 (teal)
    
    # Circle parameters
    center_x, center_y = 25, 25
    radius = 25
    
    for i, snap in enumerate(snapshots):
        if i >= len(axes):  # Safety check
            break
            
        ax = axes[i]
        
        # Determine step label
        if i == 0:
            label = 'Step 0'
        elif snapshot_steps is not None and i-1 < len(snapshot_steps):
            label = f'Step {snapshot_steps[i-1]}'
        else:
            label = f'Snapshot {i}'
        
        # Clear the axis
        ax.clear()
        
        # Plot each cell type separately with larger, more rounded appearance
        for cell_type in range(1, len(colors)):  # Skip medium (type 0)
            # Find all positions of this cell type
            positions = np.where(cell_types[snap] == cell_type)
            if len(positions[0]) > 0:
                # Use larger point size for more rounded appearance
                ax.scatter(positions[1], positions[0], 
                          c=colors[cell_type], 
                          s=point_size,  # Larger point size for rounder appearance
                          marker='o', 
                          alpha=0.9,
                          edgecolors='black',
                          linewidth=1.0)
        
        # Set axis properties for better visualization
        ax.set_xlim(center_x - radius - 2, center_x + radius + 2)
        ax.set_ylim(center_y + radius + 2, center_y - radius - 2)  # Invert y-axis to match grid coordinates
        ax.set_aspect('equal')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.2)
    
    # Hide unused subplots if there are fewer than 8 snapshots
    for i in range(num_snapshots, len(axes)):
        axes[i].set_visible(False)
        
    plt.suptitle('Potts Model Cell Sorting Progression (Rounded Scatter Plot)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_boundary_lengths(bulk_lengths, edge_lengths, interval):
    import matplotlib.pyplot as plt
    steps = [i*interval for i in range(len(bulk_lengths))]
    plt.plot(steps, bulk_lengths, label='Bulk boundary length')
    plt.plot(steps, edge_lengths, label='Edge boundary length')
    plt.xlabel('Step')
    plt.ylabel('Boundary length')
    plt.legend()
    plt.title('Boundary lengths over time')
    plt.show()

# Example usage: run the simulation and plot the results
if __name__ == "__main__":
    # Example: custom J matrix for two types (0=medium, 1, 2=cells)
    custom_J = np.array([
        [0, 16, 16],  # medium with medium, type1, type2
        [16, 2, 11],   # type1 with medium, type1, type2
        [16, 11, 14]    # type2 with medium, type1, type2
    ])
    # Run simulation and visualize snapshots
    snapshots, cell_types, snapshot_steps = run_simulation(
        size=50, n_cells=20, n_types=2, T=10.0, lam=1.0, 
        steps=1000000, snapshots=8, J_matrix=custom_J, 
        per_type_target_area=None, center_x=25, center_y=25, radius=25)
    
    # Choose which visualization to use:
    plot_snapshots_rounded(snapshots, snapshot_steps)
    # Option 2: Rounded scatter plot with larger points (uncomment to use)
    # plot_snapshots_rounded(snapshots, snapshot_steps, point_size=80)