import os
import time
import csv
import numpy as np
from PIL import Image
import multiprocessing
import concurrent.futures

"""
================================================================================
CST435 Assignment 2: Parallel Image Processing Pipeline
================================================================================
Description:
    This program implements a SEQUENTIAL Image Processing Pipeline as per assignment requirements.
    
    The Pipeline Flow (Data Dependency):
    [Input RGB] -> (1. Grayscale) -> [Gray Image] 
                -> (2. Gaussian Blur) -> [Blurred Gray]
                -> (3. Edge Detection) -> [Edge Map]
                -> (4. Sharpening) -> [Sharpened Edges]
                -> (5. Brightness) -> [Final Output]

    Benchmarks:
    1. Python `multiprocessing` module
    2. Python `concurrent.futures` module (ProcessPoolExecutor)
"""

# ==========================================
# 1. Image Filter Class (Algorithms)
# ==========================================

class ImageFilters:
    """
    Contains static methods for image processing.
    Enhanced to handle both RGB (3D) and Grayscale (2D) arrays dynamically.
    """

    @staticmethod
    def _convolve_channel(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Helper: Performs convolution on a single 2D channel."""
        channel_f = channel.astype(np.float32, copy=False)
        kernel_f = kernel.astype(np.float32, copy=False)

        rows, cols = channel_f.shape
        k_size = kernel_f.shape[0]
        pad = k_size // 2

        padded = np.pad(channel_f, pad, mode='edge')
        output = np.zeros((rows, cols), dtype=np.float32)

        # Vectorized convolution loop
        for i in range(k_size):
            for j in range(k_size):
                output += padded[i:i+rows, j:j+cols] * kernel_f[i, j]

        return output

    @staticmethod
    def apply_grayscale(img_array: np.ndarray) -> np.ndarray:
        """Filter 1: Convert RGB to Grayscale."""
        # If already grayscale (2D), return as is
        if img_array.ndim == 2:
            return img_array
            
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        gray = np.dot(img_array[..., :3].astype(np.float32), weights)
        return np.clip(gray, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_gaussian_blur(img_array: np.ndarray) -> np.ndarray:
        """Filter 2: Gaussian Blur (Supports 2D Gray and 3D RGB)."""
        kernel = (np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]], dtype=np.float32) / 16.0)
        
        # Handle Grayscale (2D)
        if img_array.ndim == 2:
            return np.clip(ImageFilters._convolve_channel(img_array, kernel), 0, 255).astype(np.uint8)
        
        # Handle RGB (3D)
        output = np.zeros_like(img_array, dtype=np.float32)
        for i in range(3):
            output[:, :, i] = ImageFilters._convolve_channel(img_array[:, :, i], kernel)  
        return np.clip(output, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_sobel(img_array: np.ndarray) -> np.ndarray:
        """Filter 3: Sobel Edge Detection (Expects Grayscale, or converts to it)."""
        # Ensure input is grayscale float
        if img_array.ndim == 3:
            gray = ImageFilters.apply_grayscale(img_array).astype(np.float32)
        else:
            gray = img_array.astype(np.float32)

        Kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
        Ky = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=np.float32)

        Gx = ImageFilters._convolve_channel(gray, Kx)
        Gy = ImageFilters._convolve_channel(gray, Ky)

        magnitude = np.sqrt(Gx**2 + Gy**2)
        return np.clip(magnitude, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_sharpen(img_array: np.ndarray) -> np.ndarray:
        """Filter 4: Sharpening (Supports 2D Gray and 3D RGB)."""
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)

        if img_array.ndim == 2:
            return np.clip(ImageFilters._convolve_channel(img_array, kernel), 0, 255).astype(np.uint8)

        output = np.zeros_like(img_array, dtype=np.float32)
        for i in range(3):
            output[:, :, i] = ImageFilters._convolve_channel(img_array[:, :, i], kernel)
        return np.clip(output, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_brightness(img_array: np.ndarray, value: int = 30) -> np.ndarray:
        """Filter 5: Brightness Adjustment."""
        out = img_array.astype(np.int16) + int(value)
        return np.clip(out, 0, 255).astype(np.uint8)

# ==========================================
# 2. Sequential Pipeline Logic
# ==========================================

def process_pipeline_task(args: tuple) -> int:
    """
    Worker function.
    Executes filters SEQUENTIALLY. Output of Step N is Input of Step N+1.
    """
    input_path, output_dir, filename, brightness_value = args

    try:
        # Load Original Image (RGB)
        with Image.open(input_path) as img:
            img = img.convert("RGB")
            original_array = np.array(img)

        # --- STEP 1: Grayscale ---
        # Input: RGB -> Output: Gray
        data_step1 = ImageFilters.apply_grayscale(original_array)
        Image.fromarray(data_step1).save(os.path.join(output_dir, f"1_gray_{filename}"))

        # --- STEP 2: Gaussian Blur ---
        # Input: Gray (from step 1) -> Output: Blurred Gray
        data_step2 = ImageFilters.apply_gaussian_blur(data_step1)
        Image.fromarray(data_step2).save(os.path.join(output_dir, f"2_blur_{filename}"))

        # --- STEP 3: Sobel Edge Detection ---
        # Input: Blurred Gray (from step 2) -> Output: Edge Map (Gray)
        data_step3 = ImageFilters.apply_sobel(data_step2)
        Image.fromarray(data_step3).save(os.path.join(output_dir, f"3_sobel_{filename}"))

        # --- STEP 4: Sharpening ---
        # Input: Edge Map (from step 3) -> Output: Sharpened Edges
        data_step4 = ImageFilters.apply_sharpen(data_step3)
        Image.fromarray(data_step4).save(os.path.join(output_dir, f"4_sharp_{filename}"))

        # --- STEP 5: Brightness Adjustment ---
        # Input: Sharpened Edges (from step 4) -> Output: Final Result
        data_step5 = ImageFilters.apply_brightness(data_step4, brightness_value)
        Image.fromarray(data_step5).save(os.path.join(output_dir, f"5_bright_{filename}"))

        return 1 # Success
    except Exception as e:
        print(f"[Error] Pipeline failed for {filename}: {e}")
        return 0

# ==========================================
# 3. Parallel Paradigm Implementations
# ==========================================

def run_paradigm_1_multiprocessing(tasks: list, num_workers: int) -> float:
    """Paradigm 1: multiprocessing.Pool"""
    start_time = time.perf_counter()
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(process_pipeline_task, tasks)
    return time.perf_counter() - start_time

def run_paradigm_2_concurrent(tasks: list, num_workers: int) -> float:
    """Paradigm 2: concurrent.futures.ProcessPoolExecutor"""
    start_time = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(process_pipeline_task, tasks))
    return time.perf_counter() - start_time

# ==========================================
# 4. Main Benchmark
# ==========================================

if __name__ == "__main__":
    # --- Config ---
    # Path handling for Windows vs Linux/GCP
    if os.name == 'nt': 
        INPUT_DIR = r"input_images" 
    else: 
        INPUT_DIR = "input_images"

    BASE_OUTPUT_DIR = "output_images"
    CSV_FILENAME = "benchmark_results.csv"
    
    MAX_IMAGES =100  # Adjust as 100
    BRIGHTNESS_VAL = 40
    WORKER_COUNTS = [1, 2, 4, 8] 

    # --- Setup ---
    print("="*85)
    print(" CST435 Assignment 2: Sequential Image Processing Pipeline")
    print("="*85)

    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR, exist_ok=True)
        print(f"[Warning] '{INPUT_DIR}' created. Please add images!")
        exit()

    all_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if MAX_IMAGES:
        all_files = all_files[:MAX_IMAGES]
    
    if not all_files:
        print(f"[Error] No images found in '{INPUT_DIR}'")
        exit()
    
    print(f"[Setup] Processing {len(all_files)} images sequentially through 5 filters.")
    print(f"[Setup] CPU Cores Available: {os.cpu_count()}")
    print("-" * 85)

    results_data = []
    base_time_mp = None
    base_time_cf = None

    # Print Header with Efficiency
    header = f"{'Workers':<8} | {'MP Time':<10} | {'CF Time':<10} | {'MP Spd':<8} | {'CF Spd':<8} | {'MP Eff':<8} | {'CF Eff':<8}"
    print(header)
    print("-" * 85)

    for w in WORKER_COUNTS:
        dir_mp = os.path.join(BASE_OUTPUT_DIR, f"mp_{w}")
        dir_cf = os.path.join(BASE_OUTPUT_DIR, f"cf_{w}")
        os.makedirs(dir_mp, exist_ok=True)
        os.makedirs(dir_cf, exist_ok=True)

        tasks_mp = [(os.path.join(INPUT_DIR, f), dir_mp, f, BRIGHTNESS_VAL) for f in all_files]
        tasks_cf = [(os.path.join(INPUT_DIR, f), dir_cf, f, BRIGHTNESS_VAL) for f in all_files]

        # Run Paradigm 1
        t_mp = run_paradigm_1_multiprocessing(tasks_mp, w)
        
        # Run Paradigm 2
        t_cf = run_paradigm_2_concurrent(tasks_cf, w)

        # Metrics Calculation
        if w == 1:
            base_time_mp = t_mp
            base_time_cf = t_cf
        #take the pool with w=1 as the serial baseline
        # Speedup = T_serial / T_parallel
        spd_mp = base_time_mp / t_mp if t_mp > 0 else 0
        spd_cf = base_time_cf / t_cf if t_cf > 0 else 0
        
        # Efficiency = Speedup / Num_Workers
        eff_mp = spd_mp / w
        eff_cf = spd_cf / w
        
        # Print Row
        print(f"{w:<8} | {t_mp:<10.4f} | {t_cf:<10.4f} | {spd_mp:<8.2f} | {spd_cf:<8.2f} | {eff_mp:<8.2f} | {eff_cf:<8.2f}")

        results_data.append({
            "Workers": w,
            "MP_Time": round(t_mp, 4), 
            "CF_Time": round(t_cf, 4),
            "MP_Speedup": round(spd_mp, 2), 
            "CF_Speedup": round(spd_cf, 2),
            "MP_Efficiency": round(eff_mp, 2),
            "CF_Efficiency": round(eff_cf, 2)
        })

    # Save Results to CSV
    with open(CSV_FILENAME, 'w', newline='') as f:
        fieldnames = ["Workers", "MP_Time", "CF_Time", "MP_Speedup", "CF_Speedup", "MP_Efficiency", "CF_Efficiency"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_data)

    print("-" * 85)
    print(f"Done. Results saved to {CSV_FILENAME}")