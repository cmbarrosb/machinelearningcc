#!/usr/bin/env python3
"""
plot_results.py

Parses training metrics from log files and plots comparisons for time and throughput.
"""

import re
import numpy as np
import matplotlib.pyplot as plt

# ---- Log files ----
SINGLE_LOG = 'logs/train_single.log'
PARALLEL_LOG = 'logs/train_parallel.log'

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
plt.figure(figsize=(6,4))
bars = plt.bar(methods, times)
ymin, ymax = min(times) * 0.98, max(times) * 1.02
plt.ylim(ymin, ymax)
plt.ylabel('Training Time (s)')
plt.title('Single-node vs DataParallel Training Time')
plt.grid(axis='y', linestyle='--', alpha=0.5)
for bar, val in zip(bars, times):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        val + (ymax - ymin) * 0.01,
        f"{val:.2f}s",
        ha='center', va='bottom'
    )
plt.savefig('time_comparison_zoom.png')
plt.close()

# ---- Plot Throughput ----
plt.figure(figsize=(6,4))
bars = plt.bar(methods, throughputs)
ymin, ymax = min(throughputs) * 0.98, max(throughputs) * 1.02
plt.ylim(ymin, ymax)
plt.ylabel('Images per Second')
plt.title('Single-node vs DataParallel Throughput')
plt.grid(axis='y', linestyle='--', alpha=0.5)
for bar, val in zip(bars, throughputs):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        val + (ymax - ymin) * 0.01,
        f"{val:.2f}",
        ha='center', va='bottom'
    )
plt.savefig('throughput_comparison_zoom.png')
plt.close()

print("Plots saved to time_comparison_zoom.png and throughput_comparison_zoom.png")