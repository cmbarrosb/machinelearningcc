

#!/usr/bin/env python3
"""
plot_results.py

Parses training metrics from log files and plots comparisons for time and throughput.
"""

import re
import matplotlib.pyplot as plt

# ---- Log files ----
SINGLE_LOG = 'train_single.log'
PARALLEL_LOG = 'train_parallel.log'

def parse_metrics(log_file):
    """Extract Time, Accuracy, Images/sec from a log file."""
    with open(log_file, 'r') as f:
        text = f.read()
    time_match = re.search(r'Time:\s*([\d.]+)s', text)
    imgs_match = re.search(r'Images/sec:\s*([\d.]+)', text)
    return float(time_match.group(1)), float(imgs_match.group(1))

# ---- Parse metrics ----
time_single, imgs_single = parse_metrics(SINGLE_LOG)
time_parallel, imgs_parallel = parse_metrics(PARALLEL_LOG)

methods = ['Single-node', 'DataParallel']
times = [time_single, time_parallel]
throughputs = [imgs_single, imgs_parallel]

# ---- Plot Training Time ----
plt.figure()
plt.bar(methods, times)
plt.ylabel('Training Time (s)')
plt.title('Single-node vs DataParallel Training Time')
plt.savefig('time_comparison.png')
plt.close()

# ---- Plot Throughput ----
plt.figure()
plt.bar(methods, throughputs)
plt.ylabel('Images per Second')
plt.title('Single-node vs DataParallel Throughput')
plt.savefig('throughput_comparison.png')
plt.close()

print("Plots saved to time_comparison.png and throughput_comparison.png")