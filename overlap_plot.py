import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    overlaps = []
    compute_times_normalized = []
    # Plotting
    plt.plot(overlaps, compute_times_normalized, marker='o')
    plt.xlabel('Overlapping Rate (1 - delay/allreduce_time)')
    plt.ylabel('Normalized Compute Time')
    plt.title('Effect of Overlapping on Compute Time')
    plt.grid(True)
    plt.show()