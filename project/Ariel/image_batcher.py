#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import numpy as np
from PIL import Image
import configparser

  # Apply the inference transforms used in the original model
from torchvision.transforms import v2
import torch 
from torchvision import datasets, transforms

train_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(256),
        v2.CenterCrop(224),
    #   v2.RandomResizedCrop(224, antialias=True),
    #   v2.RandomHorizontalFlip(p=0.5),
    #   v2.ColorJitter(0.2, 0.2, 0.2, 0.1),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])




class ImageBatcher:
    """
    Creates batches of pre-processed images.
    """

    def __init__(
        self,
        input,
        shape,
        dtype,
        max_num_images=None,
        exact_batches=False,
        config_file=None,
    ):
        """
        :param input: The input directory to read images from.
        :param shape: The tensor shape of the batch to prepare, either in NCHW or NHWC format.
        :param dtype: The (numpy) datatype to cast the batched data to.
        :param max_num_images: The maximum number of images to read from the directory.
        :param exact_batches: This defines how to handle a number of images that is not an exact multiple of the batch
        size. If false, it will pad the final batch with zeros to reach the batch size. If true, it will *remove* the
        last few images in excess of a batch size multiple, to guarantee batches are exact (useful for calibration).
        :param config_file: The path pointing to the Detectron 2 yaml file which describes the model.
        """

      

        def resnet18_setup(config_file):
            """
            Create ResNet-18 config and load dataset-specific dimensions
            from stanford_dogs.cfg.
            """
           

            dataset_cfg = configparser.ConfigParser()
            dataset_cfg.read(config_file)

            min_width = int(dataset_cfg["DIMENSIONS"]["min_width"])
            max_width = int(dataset_cfg["DIMENSIONS"]["max_width"])
            min_height = int(dataset_cfg["DIMENSIONS"]["min_height"])
            max_height = int(dataset_cfg["DIMENSIONS"]["max_height"])

            return min_width,max_width,min_height,max_height


        # ---- usage ----
        min_width,max_width,min_height,max_height = resnet18_setup(config_file)

        self.min_size_test = min(min_width,min_height)
        self.max_size_test = max(max_width,max_height)

        # Find images in the given input path.
        input = os.path.realpath(input)
        self.images = []

        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".ppm"]

        def is_image(path):
            return (
                os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions
            )

        if os.path.isdir(input):
            self.images = [
                os.path.join(input, f)
                for f in os.listdir(input)
                if is_image(os.path.join(input, f))
            ]
            self.images.sort()
        elif os.path.isfile(input):
            if is_image(input):
                self.images.append(input)
        self.num_images = len(self.images)
        if self.num_images < 1:
            print("No valid {} images found in {}".format("/".join(extensions), input))
            sys.exit(1)

        # Handle Tensor Shape.
        self.dtype = dtype
        self.shape = shape
        assert len(self.shape) == 4
        self.batch_size = shape[0]
        
        assert self.batch_size > 0
        self.format = None
        self.width = -1
        self.height = -1
        if self.shape[1] == 3:
            self.format = "NCHW"
            self.height = self.shape[2]
            self.width = self.shape[3]
        elif self.shape[3] == 3:
            self.format = "NHWC"
            self.height = self.shape[1]
            self.width = self.shape[2]
        assert all([self.format, self.width > 0, self.height > 0])

        # Adapt the number of images as needed.
        if max_num_images and 0 < max_num_images < len(self.images):
            self.num_images = max_num_images
        if exact_batches:
            self.num_images = self.batch_size * (self.num_images // self.batch_size)
        if self.num_images < 1:
            print("Not enough images to create batches")
            sys.exit(1)
        self.images = self.images[0 : self.num_images]

            
      
      

        # Subdivide the list of images into batches.
        self.num_batches = 1 + int((self.num_images - 1) / self.batch_size)
        self.batches = []
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_images)
            self.batches.append(self.images[start:end])

        # Indices.
        self.image_index = 0
        self.batch_index = 0

    def preprocess_image(self, image_path):
        """
        The image preprocessor loads an image from disk and prepares it as needed for batching. This includes padding,
        resizing, normalization, data type casting, and transposing.
        This Image Batcher implements one algorithm for now:
        * Resizes and pads the image to fit the input size.
        :param image_path: The path to the image on disk to load.
        :return: Two values: A numpy array holding the image sample, ready to be contacatenated into the rest of the
        batch, and the resize scale used, if any.
        """

        # NOTE: resize_pad logic intentionally unused.
        # Calibration uses torchvision transforms to exactly match inference.

        def resize_pad(image, pad_color=(0, 0, 0)):
            """
            A subroutine to implement padding and resizing. This will resize the image to fit fully within the input
            size, and pads the remaining bottom-right portions with the value provided.
            :param image: The PIL image object
            :pad_color: The RGB values to use for the padded area. Default: Black/Zeros.
            :return: Two values: The PIL image object already padded and cropped, and the resize scale used.
            """

            # Get characteristics.
            width, height = image.size

            # Replicates behavior of ResizeShortestEdge augmentation.
            size = self.min_size_test * 1.0
            pre_scale = size / min(height, width)
            if height < width:
                newh, neww = size, pre_scale * width
            else:
                newh, neww = pre_scale * height, size

            # If delta between min and max dimensions is so that max sized dimension reaches self.max_size_test
            # before min dimension reaches self.min_size_test, keeping the same aspect ratio. We still need to
            # maintain the same aspect ratio and keep max dimension at self.max_size_test.
            if max(newh, neww) > self.max_size_test:
                pre_scale = self.max_size_test * 1.0 / max(newh, neww)
                newh = newh * pre_scale
                neww = neww * pre_scale
            neww = int(neww + 0.5)
            newh = int(newh + 0.5)

            # Scaling factor for normalized box coordinates scaling in post-processing.
            scaling = max(newh / height, neww / width)

            # Padding.
            image = image.resize((neww, newh), resample=Image.BILINEAR)
            pad = Image.new("RGB", (self.width, self.height))
            pad.paste(pad_color, [0, 0, self.width, self.height])
            pad.paste(image)
            return pad, scaling

        scale = None
        image = Image.open(image_path)
        image = image.convert(mode="RGB")

        image=train_transform(image)
        image=image.numpy()

        return image, None
        

        


        
        
        # Pad with mean values of STANFORD DOGS dataset, since padding is applied before actual model's
        # preprocessor steps (Sub, Div ops), we need to pad with mean values in order to reverse
        # the effects of Sub and Div, so that padding after model's preprocessor will be with actual 0s.
      #  image, scale = resize_pad(image, (0.485,0.456,0.406))
       # image = np.asarray(image, dtype=np.float32)
        # Change HWC -> CHW.
       # image = np.transpose(image, (2, 0, 1))
        # Change RGB -> BGR.
        #return image[[2, 1, 0]], scale

    def get_batch(self):
        """
        Retrieve the batches. This is a generator object, so you can use it within a loop as:
        for batch, images in batcher.get_batch():
           ...
        Or outside of a batch with the next() function.
        :return: A generator yielding three items per iteration: a numpy array holding a batch of images, the list of
        paths to the images loaded within this batch, and the list of resize scales for each image in the batch.
        """
        for i, batch_images in enumerate(self.batches):
            batch_data = np.zeros(self.shape, dtype=self.dtype)
            batch_scales = [None] * len(batch_images)
            for i, image in enumerate(batch_images):
                self.image_index += 1
                batch_data[i], batch_scales[i] = self.preprocess_image(image)
            self.batch_index += 1
            
            
            assert batch_data.dtype == np.float32
            assert batch_data.shape == (self.batch_size, 3, 224, 224)
            assert np.isfinite(batch_data).all()

            yield batch_data, batch_images, batch_scales


      