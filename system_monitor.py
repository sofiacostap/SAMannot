""" Change between original version and this is the addition of  error handling for systems without an NVIDIA GPU.
 original: blindly assumes `pynvml` is installed and a GPU is present, which will crash on a CPU-only machine. 
 this version: wraps all GPU-related tasks in `try/except` blocks and checks a global boolean flag (`GPU_AVAILABLE`).

The UI code (`SystemMonitorWindow`) remains completely unchanged structurally, aside from minor whitespace and code-formatting updates.


 1. Imports & Global Guard Flag

* original: Directly imports from `pynvml` at the top level. If the module is missing, the script crashes instantly.
* this version: Moves the `pynvml` import inside a `try/except` block. It introduces a `GPU_AVAILABLE` flag set to `True` or `False` 
depending on whether the import and initial `nvmlInit()` succeed.

 2. `SystemMonitor.__init__`

* original: Explicitly calls `nvmlInit()` and queries `nvmlDeviceGetCount()` right away. It iterates through devices unconditionally.
* this version: Explicitly initializes `self.device_count = 0` up front.
* Wraps the device querying and iteration inside an `if GPU_AVAILABLE:` condition block, further protected by an inner `try/except` block
 to catch unexpected runtime initialization errors.



 3. `SystemMonitor.update_system_info`

* original: Iterates through the GPU indices to fetch utilization metrics and VRAM stats without any checks.
* this version: Places the GPU looping mechanism inside an `if GPU_AVAILABLE:` guard and swallows any potential exceptions using a silent `except Exception: pass`.

 4. `SystemMonitor.__del__`

* original: Blindly calls `nvmlShutdown()`.
* this version: Only calls `nvmlShutdown()` if `GPU_AVAILABLE` is true, and safely wraps it in a `try/except` block to prevent exceptions during script termination."""

import psutil
import tkinter as tk
from visual_widget_utils import *

# Try to import GPU monitoring — gracefully skip if no NVIDIA GPU available
GPU_AVAILABLE = False
try:
    from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlDeviceGetName
    nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    print("No NVIDIA GPU detected — running in CPU-only mode (annotation still works, propagation disabled)")

class SystemMonitor:

    def __init__(self):
        self.device_count = 0
        self.device_handles = []
        self.device_names = []
        self.used_ram = 0
        self.total_ram = 0
        self.gpu_util = []
        self.gpu_memory = []

        if GPU_AVAILABLE:
            try:
                self.device_count = nvmlDeviceGetCount()
                for idx in range(self.device_count):
                    handle = nvmlDeviceGetHandleByIndex(idx)
                    self.device_handles.append(handle)
                    self.device_names.append(nvmlDeviceGetName(handle))
                    self.gpu_memory.append(0)
                    self.gpu_util.append(0)
            except Exception as e:
                print(f"GPU init warning: {e}")

        self.update_system_info()

    def update_system_info(self):
        virtual_memory = psutil.virtual_memory()
        self.used_ram = virtual_memory.used
        self.total_ram = virtual_memory.total

        if GPU_AVAILABLE:
            try:
                for idx in range(self.device_count):
                    self.gpu_util[idx] = nvmlDeviceGetUtilizationRates(self.device_handles[idx]).gpu
                    gpu_mem = nvmlDeviceGetMemoryInfo(self.device_handles[idx])
                    self.gpu_memory[idx] = (gpu_mem.used, gpu_mem.total)
            except Exception:
                pass

    def __del__(self):
        if GPU_AVAILABLE:
            try:
                nvmlShutdown()
            except Exception:
                pass


class SystemMonitorWindow(tk.Toplevel):

    def __init__(self, parent):
        super().__init__(parent)
        self.configure(bg="#f5f6f7")
        self.title("System Resources")

        self.info_frame = tk.Frame(self, bg="#f5f5f5")
        self.info_frame.pack(fill=tk.BOTH, padx=10, pady=10)

        self.is_open = True

        # RAM
        ram_frame = tk.Frame(self.info_frame, bg="#f5f5f5")
        ram_frame.grid(row=0, column=0, padx=15)
        ram_label = tk.Label(ram_frame, text="RAM", bg="#f5f5f5", fg="#222222", font=("Segoe UI", 14, "bold"))
        ram_label.pack(pady=(0, 5))
        self.circular_progress_bar_ram = CircularProgress(self.info_frame, size=200)
        self.circular_progress_bar_ram.grid(row=1, column=0, sticky="ew")

        # GPU Util
        gpu_frame = tk.Frame(self.info_frame, bg="#f5f5f5")
        gpu_frame.grid(row=0, column=1, padx=15)
        gpu_label = tk.Label(gpu_frame, text="GPU", bg="#f5f5f5", fg="#222222", font=("Segoe UI", 14, "bold"))
        gpu_label.pack(pady=(0, 5))
        self.circular_progress_bar_gpu = CircularProgress(self.info_frame, size=200)
        self.circular_progress_bar_gpu.grid(row=1, column=1, sticky="ew")

        # VRAM
        vram_frame = tk.Frame(self.info_frame, bg="#f5f5f5")
        vram_frame.grid(row=0, column=2, padx=15)
        vram_label = tk.Label(vram_frame, text="VRAM", bg="#f5f5f5", fg="#222222", font=("Segoe UI", 14, "bold"))
        vram_label.pack(pady=(0, 5))
        self.circular_progress_bar_vram = CircularProgress(self.info_frame, size=200)
        self.circular_progress_bar_vram.grid(row=1, column=2, sticky="ew")

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def hide(self):
        self.withdraw()
        self.is_open = False

    def show(self):
        self.deiconify()
        self.lift()
        self.focus_force()
        self.is_open = True

    def on_close(self):
        self.hide()