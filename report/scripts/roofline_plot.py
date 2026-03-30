#!/usr/bin/env python3
"""Generate a roofline plot for H100 80GB HBM3 with kernel data points."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# H100 80GB HBM3 specs (SXM variant, FP32 CUDA cores only)
peak_compute = 67.0     # TFLOP/s FP32 (CUDA cores, non-tensor)
peak_bandwidth = 3.35    # TB/s

# Ridge point
ridge_ai = peak_compute / peak_bandwidth  # ~20 FLOPs/byte
print(f"Ridge point: {ridge_ai:.1f} FLOPs/byte")

# AI range for plot
ai = np.logspace(-2, 3, 500)

# Roofline: min(peak_compute, peak_bandwidth * ai)
roofline = np.minimum(peak_compute, peak_bandwidth * ai)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.loglog(ai, roofline * 1e3, 'k-', linewidth=2, label='Roofline (H100 FP32)')

# Fill regions
mem_bound = ai < ridge_ai
ax.fill_between(ai[mem_bound], roofline[mem_bound] * 1e3, 0.01,
                alpha=0.05, color='blue')
comp_bound = ai >= ridge_ai
ax.fill_between(ai[comp_bound], roofline[comp_bound] * 1e3, 0.01,
                alpha=0.05, color='red')

# Mark ridge point
ax.axvline(x=ridge_ai, color='gray', linestyle='--', alpha=0.5)
ax.annotate(f'Ridge point\n({ridge_ai:.1f} FLOPs/byte)',
            xy=(ridge_ai, peak_compute * 1e3),
            xytext=(ridge_ai * 3, peak_compute * 0.3e3),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=9, color='gray')

# Kernel data points from benchmark
# (AI in FLOPs/byte, achieved GFLOPs/s)
kernels = {
    'Gaussian Blur\n(shared mem tiling)': (24.50, 1460.86),
    'Sobel\n(global mem)':                (0.40, 144.54),
    'Hist+Equalise':                      (1.33, 11.03),
}

colors = {'Gaussian Blur\n(shared mem tiling)': 'blue',
          'Sobel\n(global mem)': 'green',
          'Hist+Equalise': 'red'}
markers = {'Gaussian Blur\n(shared mem tiling)': 'o',
           'Sobel\n(global mem)': 's',
           'Hist+Equalise': '^'}

for name, (ai_val, gflops) in kernels.items():
    ax.plot(ai_val, gflops, markers[name], color=colors[name],
            markersize=10, label=f'{name} (AI={ai_val:.2f})', zorder=5)

# Labels
ax.set_xlabel('Arithmetic Intensity (FLOPs/byte)', fontsize=12)
ax.set_ylabel('Performance (GFLOPs/s)', fontsize=12)
ax.set_title('Roofline Model — NVIDIA H100 80GB HBM3 (FP32 CUDA Cores)', fontsize=13)
ax.legend(loc='lower right', fontsize=8)
ax.set_xlim(0.01, 1000)
ax.set_ylim(1, 200000)
ax.grid(True, which='both', alpha=0.3)

# Add bandwidth and compute annotations
ax.text(0.05, peak_bandwidth * 0.05 * 1e3 * 0.6, 'Memory\nBound',
        fontsize=10, color='blue', alpha=0.5, ha='center')
ax.text(200, peak_compute * 0.5e3, 'Compute\nBound',
        fontsize=10, color='red', alpha=0.5, ha='center')

plt.tight_layout()
plt.savefig('assets/roofline.png', dpi=200, bbox_inches='tight')
plt.savefig('assets/roofline.pdf', bbox_inches='tight')
print("Saved to assets/roofline.png and assets/roofline.pdf")
