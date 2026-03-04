
import os
import sys
import logging
import argparse

import numpy as np

import tensorrt as trt
from cuda import cudart

from .image_batcher import ImageBatcher

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")

import common
sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


class EngineCalibrator(trt.IInt8MinMaxCalibrator):
    """
    Implements the INT8 MinMax Calibrator.
    """

    def __init__(self, cache_file):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.cache_file = cache_file
        self.image_batcher = None
        self.batch_allocation = None
        self.batch_generator = None

    def set_image_batcher(self, image_batcher: ImageBatcher):
        """
        Define the image batcher to use, if any. If using only the cache file, an image batcher doesn't need
        to be defined.
        :param image_batcher: The ImageBatcher object
        """
        self.image_batcher = image_batcher
     

        self.size = int(
            np.dtype(self.image_batcher.dtype).itemsize
            * np.prod(self.image_batcher.shape)
        )
        self.batch_allocation = common.cuda_call(cudart.cudaMalloc(self.size))
        self.batch_generator = self.image_batcher.get_batch()

    def get_batch_size(self):
        """
        Overrides from trt.IInt8MinMaxCalibrator.
        Get the batch size to use for calibration.
        :return: Batch size.
        """
        if self.image_batcher:
            return self.image_batcher.batch_size
        return 1

    def get_batch(self, names):
        """
        Overrides from trt.IInt8MinMaxCalibrator.
        Get the next batch to use for calibration, as a list of device memory pointers.
        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: A list of int-casted memory pointers.
        """
        if not self.image_batcher:
            return None
        try:
            batch, _, _ = next(self.batch_generator)
            log.info(
                "Calibrating image {} / {}".format(
                    self.image_batcher.image_index, self.image_batcher.num_images
                )
            )
            common.memcpy_host_to_device(
                self.batch_allocation, np.ascontiguousarray(batch)
            )

            return [int(self.batch_allocation)]
        except StopIteration:
            log.info("Finished calibration batches")
            return None

    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8MinMaxCalibrator.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                log.info("Using calibration cache file: {}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        """
        Overrides from trt.IInt8MinMaxCalibrator.
        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        if self.cache_file is None:
            return
        with open(self.cache_file, "wb") as f:
            log.info("Writing calibration cache data to: {}".format(self.cache_file))
            f.write(cache)

