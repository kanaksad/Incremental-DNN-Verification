# import torch
# import matplotlib.pyplot as plt

# # Paths to the files
# repaird_lbs_ubs_path = 'results/nnverify-nets-aprnn-repaired-mnist_relu_9_100_polytope.pthDomain.DEEPPOLY.pt'
# unrepaird_lbs_ubs_path = 'results/nnverify-nets-aprnn-unrepaired-mnist_relu_9_100.pthDomain.DEEPPOLY.pt'

# # Load the data
# repaird_lbs_ubs = torch.load(repaird_lbs_ubs_path)
# unrepaird_lbs_ubs = torch.load(unrepaird_lbs_ubs_path)

# # Extract the lists of lower and upper bounds
# unrepaired_lbs = unrepaird_lbs_ubs['lbs']
# unrepaired_ubs = unrepaird_lbs_ubs['ubs']

# repaired_lbs = repaird_lbs_ubs['lbs']
# repaired_ubs = repaird_lbs_ubs['ubs']

# # Initialize empty lists to collect all the bounds
# all_unrepaired_lbs = []
# all_unrepaired_ubs = []
# all_repaired_lbs = []
# all_repaired_ubs = []

# # Iterate over each set of bounds and combine them
# for i, (unrepaired_lb, unrepaired_ub, repaired_lb, repaired_ub) in enumerate(zip(unrepaired_lbs, unrepaired_ubs, repaired_lbs, repaired_ubs)):
#     print(f"\nProcessing bounds for proof {i+1}:")
    
#     # Check if any tensors are empty and skip if necessary
#     if unrepaired_lb.numel() == 0 or unrepaired_ub.numel() == 0 or repaired_lb.numel() == 0 or repaired_ub.numel() == 0:
#         print("One or more tensors are empty, skipping comparison.")
#         continue

#     # Ensure all tensors have the same shape
#     if unrepaired_lb.shape == unrepaired_ub.shape == repaired_lb.shape == repaired_ub.shape:
#         # Flatten tensors and detach them before converting to numpy
#         all_unrepaired_lbs.extend(unrepaired_lb.detach().view(-1).numpy())
#         all_unrepaired_ubs.extend(unrepaired_ub.detach().view(-1).numpy())
#         all_repaired_lbs.extend(repaired_lb.detach().view(-1).numpy())
#         all_repaired_ubs.extend(repaired_ub.detach().view(-1).numpy())
#     else:
#         print(f"Tensors in set {i+1} do not have the same shape, skipping comparison.")

# # Convert lists to numpy arrays (optional step if needed for performance)
# all_unrepaired_lbs = torch.tensor(all_unrepaired_lbs).numpy()
# all_unrepaired_ubs = torch.tensor(all_unrepaired_ubs).numpy()
# all_repaired_lbs = torch.tensor(all_repaired_lbs).numpy()
# all_repaired_ubs = torch.tensor(all_repaired_ubs).numpy()

# # Create a plot for visualizing the combined bounds
# plt.figure(figsize=(12, 8))

# # Number of elements in the combined arrays
# elements = range(len(all_unrepaired_lbs))

# # Plot combined bounds
# plt.plot(elements, all_unrepaired_lbs, 'b--', label='Unrepaired LB')
# plt.plot(elements, all_unrepaired_ubs, 'r-', label='Unrepaired UB')
# plt.plot(elements, all_repaired_lbs, 'y--', label='Repaired LB')
# plt.plot(elements, all_repaired_ubs, 'g-', label='Repaired UB')

# # Add labels, title, and legend
# plt.xlabel('Element Index')
# plt.ylabel('Bound Value')
# plt.title('Combined Comparison of Unrepaired vs Repaired Bounds Across All Proofs')
# plt.legend()

# # Save the figure as an image
# plt.savefig('comparison_combined_bounds.png')

# # Show the plot
# plt.show()


# # Create subplots for visualizing bounds separately
# fig, axs = plt.subplots(2, 1, figsize=(12, 12))

# # Plot lower bounds comparison
# axs[0].plot(elements, all_unrepaired_lbs, 'b--', label='Unrepaired LB')
# axs[0].plot(elements, all_repaired_lbs, 'y--', label='Repaired LB')
# axs[0].set_title('Lower Bound Comparison')
# axs[0].set_xlabel('Element Index')
# axs[0].set_ylabel('Bound Value')
# axs[0].legend()

# # Plot upper bounds comparison
# axs[1].plot(elements, all_unrepaired_ubs, 'r-', label='Unrepaired UB')
# axs[1].plot(elements, all_repaired_ubs, 'g-', label='Repaired UB')
# axs[1].set_title('Upper Bound Comparison')
# axs[1].set_xlabel('Element Index')
# axs[1].set_ylabel('Bound Value')
# axs[1].legend()

# # Save the figure as an image
# plt.savefig('comparison_combined_bounds_subplots.png')

# # Show the plot
# plt.show()

# # Create a scatter plot for visualizing the combined bounds
# plt.figure(figsize=(12, 8))

# # Plot combined bounds using scatter
# plt.scatter(elements, all_unrepaired_lbs, color='blue', marker='x', alpha=0.3, label='Unrepaired LB')
# plt.scatter(elements, all_unrepaired_ubs, color='red', marker='o', alpha=0.3, label='Unrepaired UB')
# plt.scatter(elements, all_repaired_lbs, color='yellow', marker='x', alpha=0.3, label='Repaired LB')
# plt.scatter(elements, all_repaired_ubs, color='green', marker='o', alpha=0.3, label='Repaired UB')

# # Add labels, title, and legend
# plt.xlabel('Element Index')
# plt.ylabel('Bound Value')
# plt.title('Combined Comparison of Unrepaired vs Repaired Bounds (Scatter Plot)')
# plt.legend()

# # Save the figure as an image
# plt.savefig('comparison_combined_bounds_scatter.png')

# # Show the plot
# plt.show()


import torch
import matplotlib.pyplot as plt

# Paths to the files
repaird_lbs_ubs_path = 'results/500-nnverify-nets-aprnn-repaired-mnist_relu_9_100_polytopek16.pthDomain.DEEPPOLY.pt'
unrepaird_lbs_ubs_path = 'results/500-nnverify-nets-aprnn-unrepaired-mnist_relu_9_100.pthDomain.DEEPPOLY.pt'

# Load the data
repaird_lbs_ubs = torch.load(repaird_lbs_ubs_path)
unrepaird_lbs_ubs = torch.load(unrepaird_lbs_ubs_path)

# Extract the lists of lower and upper bounds
unrepaired_lbs = unrepaird_lbs_ubs['lbs']
unrepaired_ubs = unrepaird_lbs_ubs['ubs']

repaired_lbs = repaird_lbs_ubs['lbs']
repaired_ubs = repaird_lbs_ubs['ubs']

# Initialize empty lists to collect all the bounds
all_unrepaired_lbs = []
all_unrepaired_ubs = []
all_repaired_lbs = []
all_repaired_ubs = []

# Iterate over each set of bounds and combine them
for i, (unrepaired_lb, unrepaired_ub, repaired_lb, repaired_ub) in enumerate(zip(unrepaired_lbs, unrepaired_ubs, repaired_lbs, repaired_ubs)):
    print(f"\nProcessing bounds for proof {i+1}:")
    
    # Check if any tensors are empty and skip if necessary
    if unrepaired_lb.numel() == 0 or unrepaired_ub.numel() == 0 or repaired_lb.numel() == 0 or repaired_ub.numel() == 0:
        print("One or more tensors are empty, skipping comparison.")
        continue

    # Ensure all tensors have the same shape
    if unrepaired_lb.shape == unrepaired_ub.shape == repaired_lb.shape == repaired_ub.shape:
        # Flatten tensors and detach them before converting to numpy
        all_unrepaired_lbs.extend(unrepaired_lb.detach().view(-1).numpy())
        all_unrepaired_ubs.extend(unrepaired_ub.detach().view(-1).numpy())
        all_repaired_lbs.extend(repaired_lb.detach().view(-1).numpy())
        all_repaired_ubs.extend(repaired_ub.detach().view(-1).numpy())
    else:
        print(f"Tensors in set {i+1} do not have the same shape, skipping comparison.")

# Convert lists to numpy arrays (optional step if needed for performance)
all_unrepaired_lbs = torch.tensor(all_unrepaired_lbs).numpy()
all_unrepaired_ubs = torch.tensor(all_unrepaired_ubs).numpy()
all_repaired_lbs = torch.tensor(all_repaired_lbs).numpy()
all_repaired_ubs = torch.tensor(all_repaired_ubs).numpy()

# Initialize counters for containment, overlap, and non-overlap checks
contained_in_unrepaired = 0
contained_in_repaired = 0
overlapping_c = 0
no_overlap = 0
equals = 0

# Check containment, overlap, and non-overlap for each element
for i in range(len(all_unrepaired_lbs)):
    # Check if repaired bound is contained within unrepaired bound
    if all_unrepaired_lbs[i] == all_repaired_lbs[i] and all_repaired_ubs[i] == all_unrepaired_ubs[i]:
        equals += 1
        continue
    
    if all_unrepaired_lbs[i] <= all_repaired_lbs[i] and all_repaired_ubs[i] <= all_unrepaired_ubs[i]:
        contained_in_unrepaired += 1
        continue

    # Check if unrepaired bound is contained within repaired bound
    if all_repaired_lbs[i] <= all_unrepaired_lbs[i] and all_unrepaired_ubs[i] <= all_repaired_ubs[i]:
        contained_in_repaired += 1
        continue

    # Check if bounds are overlapping
    if (all_unrepaired_lbs[i] <= all_repaired_ubs[i] and all_unrepaired_ubs[i] >= all_repaired_lbs[i]):
        overlapping_c += 1
        continue

    # Check if bounds are not overlapping at all
    if (all_repaired_ubs[i] < all_unrepaired_lbs[i] or all_repaired_lbs[i] > all_unrepaired_ubs[i]):
        no_overlap += 1

# Initialize counters for containment, overlap, and non-overlap checks
overlapping = 0

# Create a scatter plot with connecting lines for visualizing the combined bounds
plt.figure(figsize=(12, 8))

# Number of elements in the combined arrays
elements = range(len(all_unrepaired_lbs))

# Plot unrepaired bounds with lines connecting LB and UB (red color)
for i in elements:
    plt.plot([i, i], [all_unrepaired_lbs[i], all_unrepaired_ubs[i]], color='red', linestyle='-', alpha=1, label='Unrepaired Bound' if i == 0 else "")

# Plot repaired bounds with lines connecting LB and UB (green color)
for i in elements:
    plt.plot([i, i], [all_repaired_lbs[i], all_repaired_ubs[i]], color='green', linestyle='-', alpha=1, label='Repaired Bound' if i == 0 else "")

# Check containment, overlap, and non-overlap for each element and plot the overlapping part in blue
for i in elements:
    if (all_unrepaired_lbs[i] <= all_repaired_ubs[i] and all_unrepaired_ubs[i] >= all_repaired_lbs[i]):
        overlapping += 1
        # Plot overlapping bounds in blue
        plt.plot([i, i], [max(all_unrepaired_lbs[i], all_repaired_lbs[i]), min(all_unrepaired_ubs[i], all_repaired_ubs[i])], color='blue', linestyle='-', alpha=0.4, label='Overlapping Bound' if i == 0 else "")

# Add labels, title, and legend
plt.xlabel('Index')
plt.ylabel('Bound')
plt.legend()

# Add the counts to the plot as annotations
plt.text(len(elements) * 0.1, max(all_unrepaired_ubs), f"Total: {len(all_unrepaired_lbs)}", fontsize=8, color='black')
plt.text(len(elements) * 0.1, max(all_unrepaired_ubs) * 0.95, f"Repaired Contained in Unrepaired: {contained_in_unrepaired}", fontsize=8, color='black')
plt.text(len(elements) * 0.1, max(all_unrepaired_ubs) * 0.90, f"Unrepaired Contained in Repaired: {contained_in_repaired}", fontsize=8, color='black')
plt.text(len(elements) * 0.1, max(all_unrepaired_ubs) * 0.85, f"Overlapping: {overlapping_c}", fontsize=8, color='black')
plt.text(len(elements) * 0.1, max(all_unrepaired_ubs) * 0.80, f"Non-overlapping: {no_overlap}", fontsize=8, color='black')
plt.text(len(elements) * 0.1, max(all_unrepaired_ubs) * 0.75, f"Same: {equals}", fontsize=8, color='black')

# Save the figure as an image
plt.savefig('comparison_scatter_with_lines_and_overlaps.png')

# Show the plot
plt.show()

# Print the results in the console as well
print(f"Number of overlapping bounds: {overlapping}")

