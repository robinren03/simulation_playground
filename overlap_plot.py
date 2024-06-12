import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    overlaps = [0, 0.25, 0.5, 0.75, 1]
    compute_times_normalized = [0.9734838637571203, 0.975997420515524, 0.9806217117351831, 0.9814013453938705, 0.9848868931531025]
    compute_times_normalized = [a/compute_times_normalized[0] for a in compute_times_normalized]
    # Plotting
    plt.plot(overlaps, compute_times_normalized, marker='o')
    plt.xlabel('Overlapping Rate (1 - delay/allreduce_time)')
    plt.ylabel('Normalized Compute Time')
    plt.title('Effect of Overlapping on Compute Time')
    plt.grid(True)
    plt.savefig("overlap_plot.png")