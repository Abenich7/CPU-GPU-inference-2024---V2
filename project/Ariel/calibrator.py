import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os

class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, dataloader, cache_file="calib.cache"):
        super().__init__()
        self.dataloader = dataloader
        self.cache_file = cache_file
        self.iterator = iter(dataloader)

        self.batch_size = dataloader.batch_size
        sample = next(self.iterator)[0]
        self.device_input = cuda.mem_alloc(sample.numpy().nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        try:
            batch = next(self.iterator)[0].numpy()
            cuda.memcpy_htod(self.device_input, batch)
            return [int(self.device_input)]
        except StopIteration:
            return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
