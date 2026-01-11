# CST435 Assignment 2: Parallel Image Processing

## 1. Project Overview
This project implements a parallel image processing pipeline using Python on Google Cloud Platform (GCP). It processes the Food-101 dataset using a custom 5-stage filter (Grayscale, Blur, Edge Detection, Sharpen, Brightness).

The system compares two parallel paradigms to demonstrate speedup on multi-core vCPUs:
* **Paradigm 1**: `multiprocessing` module (Manual process management)
* **Paradigm 2**: `concurrent.futures.ProcessPoolExecutor` (High-level abstraction)

---

## 2. How to Run on GCP

Follow these steps to deploy and execute the project on a Google Cloud Platform (GCP) Compute Engine instance.

### 📋 Prerequisites
* **GCP VM Instance**: Recommended `e2-standard-4` (4 vCPUs, 16GB RAM) or higher.
* **OS**: Ubuntu 20.04 LTS / Debian 10+ (Python 3.8+ pre-installed).

### ⚙️ Step-by-Step Guide

#### Step 1: Initialize Environment
Connect to your GCP VM via SSH and update the system packages:
```bash
sudo apt update
sudo apt install git python3-pip htop -y
```
#### Step 2: Clone Repository
Clone the project repository to your VM:
```bash
git clone [https://github.com/qiaohaoyan777/CST435-Assignment-2.git](https://github.com/qiaohaoyan777/CST435-Assignment-2.git)
cd CST435-Assignment-2
```
#### Step 3: Install Dependencies
Install the required Python libraries (NumPy, Pillow, Matplotlib):
```bash
pip3 install -r requirements.txt
# Alternatively: pip3 install numpy pillow matplotlib
```
#### Step 4: Verify Input Data
Ensure the input directory contains the Food-101 dataset images.
```bash
ls input/
# If empty, upload images or use wget to download a sample dataset.
```
#### Step 5: Run Benchmark
Execute the main script to start the parallel processing benchmark:
```bash
python3 main.py
```

## 3. Results Summary (GCP e2-standard-4)

| Workers | MP Time (s) | CF Time (s) | MP Speedup | CF Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **1** | 2.61 | 2.59 | 1.00x | 1.00x |
| **2** | 1.54 | 1.50 | 1.69x | 1.73x |
| **4** | 1.20 | 1.18 | 2.17x | 2.20x |
| **8** | 1.28 | 1.26 | 2.04x | 2.05x |

**Conclusion**: 
* **Peak Performance**: Both paradigms achieved maximum speedup at **4 workers** (~2.20x), which aligns perfectly with the 4 vCPUs available on the `e2-standard-4` instance.
* **Paradigm Comparison**: **Concurrent Futures (CF)** performed consistently faster than Multiprocessing (MP) across all tests (e.g., 1.18s vs 1.20s at peak), likely due to its optimized internal task scheduling.
* **Overhead**: Increasing workers to 8 (beyond physical cores) caused a slight performance drop due to context switching overhead, validating the importance of matching worker count to CPU cores.
