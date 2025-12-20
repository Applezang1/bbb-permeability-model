import numpy as np
import torch

# Set print size of PyTorch output
np.set_printoptions(precision=3, floatmode="fixed")
torch.set_printoptions(precision=3)

# Perform convolution using PyTorch
def conv_pytorch(image, conv_weights, stride=1, pad =1):
  # Convert image and kernel to tensors
  image_tensor = torch.from_numpy(image) # (batchSize, channelsIn, imageHeightIn, =imageWidthIn)
  conv_weights_tensor = torch.from_numpy(conv_weights) # (channelsOut, channelsIn, kernelHeight, kernelWidth)
  # Undergo convolution
  output_tensor = torch.nn.functional.conv2d(image_tensor, conv_weights_tensor, stride=stride, padding=pad)
  # Convert back from PyTorch and return 
  return(output_tensor.numpy()) # (batchSize channelsOut imageHeightOut imageHeightIn)

# Perform convolution in Numpy
def conv_numpy_1(image, weights, pad=1):
    # Perform zero padding
    if pad != 0:
        image = np.pad(image, ((0, 0), (0 ,0), (pad, pad), (pad, pad)),'constant')

    # Determine sizes of image array and kernel weights
    batchSize,  channelsIn, imageHeightIn, imageWidthIn = image.shape
    channelsOut, channelsIn, kernelHeight, kernelWidth = weights.shape

    # Determine size of output arrays
    imageHeightOut = np.floor(1 + imageHeightIn - kernelHeight).astype(int)
    imageWidthOut = np.floor(1 + imageWidthIn - kernelWidth).astype(int)

    # Define an array structure for the output of the CNN
    out = np.zeros((batchSize, channelsOut, imageHeightOut, imageWidthOut), dtype=np.float32)

    # For the location of each pixel on the final image
    for c_y in range(imageHeightOut):
      for c_x in range(imageWidthOut):
        # For the location of each pixel on the kernel
        for c_kernel_y in range(kernelHeight):
          for c_kernel_x in range(kernelWidth):
            # Compute the value of the pixel and the value of the corresponding kernel weight
            this_pixel_value = image[0, 0, c_kernel_y+c_y, c_kernel_x+c_x]
            this_weight = weights[0, 0, c_kernel_y, c_kernel_x]

            # Multiply the value of the kernel by the value at the location of the pixel to compute the activation map
            out[0, 0, c_y, c_x] += np.sum(this_pixel_value * this_weight)

    return out

# Initialize Hyperparameters
np.random.seed(1)
n_batch = 1
image_height = 4
image_width = 6
channels_in = 1
kernel_size = 3
channels_out = 1

# Define an input image
input_image= np.random.normal(size=(n_batch, channels_in, image_height, image_width))

# Initialize convolution kernel weights
conv_weights = np.random.normal(size=(channels_out, channels_in, kernel_size, kernel_size))

# Perform convolution using PyTorch
conv_results_pytorch = conv_pytorch(input_image, conv_weights, stride=1, pad=1)
print("PyTorch Results")
print(conv_results_pytorch)

# Perform convolution in numpy
print("Numpy results")
conv_results_numpy = conv_numpy_1(input_image, conv_weights)
print(conv_results_numpy)

# Perform Convolution in Numpy (with stride)
def conv_numpy_2(image, weights, stride=1, pad=1):

    # Perform zero padding
    if pad != 0:
        image = np.pad(image, ((0, 0), (0 ,0), (pad, pad), (pad, pad)),'constant')

    # Determine sizes of image array and kernel weights
    batchSize,  channelsIn, imageHeightIn, imageWidthIn = image.shape
    channelsOut, channelsIn, kernelHeight, kernelWidth = weights.shape

    # Determine size of output arrays
    imageHeightOut = np.floor(1 + (imageHeightIn - kernelHeight) / stride).astype(int)
    imageWidthOut = np.floor(1 + (imageWidthIn - kernelWidth) / stride).astype(int)

    # Define an array structure for the output of the CNN
    out = np.zeros((batchSize, channelsOut, imageHeightOut, imageWidthOut), dtype=np.float32)

    # For the location of each pixel on the final image
    for c_y in range(imageHeightOut):
      for c_x in range(imageWidthOut):
        # For the location of each pixel on the kernel
        for c_kernel_y in range(kernelHeight):
          for c_kernel_x in range(kernelWidth):
            # Compute the value of the pixel and the value of the corresponding kernel weight
            # Incorporate stride towards the location of the pixel
            this_pixel_value = image[0, 0, c_kernel_y+c_y*stride, c_kernel_x+c_x*stride]
            this_weight = weights[0, 0, c_kernel_y, c_kernel_x]

            # Multiply the value of the kernel by the value at the location of the pixel to compute the activation map
            out[0, 0, c_y, c_x] += np.sum(this_pixel_value * this_weight)

    return out

# Initialize Hyperparameters
np.random.seed(1)
n_batch = 1
image_height = 12
image_width = 10
channels_in = 1
kernel_size = 3
channels_out = 1
stride = 2

# Define an input image
input_image= np.random.normal(size=(n_batch, channels_in, image_height, image_width))

# Initialize convolution kernel weights
conv_weights = np.random.normal(size=(channels_out, channels_in, kernel_size, kernel_size))

# Perform convolution using PyTorch
conv_results_pytorch = conv_pytorch(input_image, conv_weights, stride, pad=1)
print("PyTorch Results")
print(conv_results_pytorch)

# Perform convolution in Numpy
print("Numpy results")
conv_results_numpy = conv_numpy_2(input_image, conv_weights, stride, pad=1)
print(conv_results_numpy)

# Perform convolution in Numpy (with input and output layers)
def conv_numpy_3(image, weights, stride=1, pad=1):
    # Perform zero padding
    if pad != 0:
        image = np.pad(image, ((0, 0), (0 ,0), (pad, pad), (pad, pad)),'constant')

    # Determine sizes of image array and kernel weights
    batchSize,  channelsIn, imageHeightIn, imageWidthIn = image.shape
    channelsOut, channelsIn, kernelHeight, kernelWidth = weights.shape

    # Determine size of output arrays
    imageHeightOut = np.floor(1 + (imageHeightIn - kernelHeight) / stride).astype(int)
    imageWidthOut = np.floor(1 + (imageWidthIn - kernelWidth) / stride).astype(int)

    # Define an array structure for the output of the CNN
    out = np.zeros((batchSize, channelsOut, imageHeightOut, imageWidthOut), dtype=np.float32)

    # For the location of each pixel on the final image
    for c_y in range(imageHeightOut):
      for c_x in range(imageWidthOut):
        # For each input layer and output layer
        for c_channel_out in range(channelsOut):
          for c_channel_in in range(channelsIn):
            # For the location of each pixel on the kernel
            for c_kernel_y in range(kernelHeight):
              for c_kernel_x in range(kernelWidth):
                  # Compute the value of the pixel and the value of the corresponding kernel for each input and output layer
                  this_pixel_value = image[0, c_channel_in, c_kernel_y+c_y, c_kernel_x+c_x]
                  this_weight = weights[c_channel_out, c_channel_in, c_kernel_y, c_kernel_x]

                  # Multiply the value of the kernel by the value at the location of the pixel to compute the activation map
                  out[0, c_channel_out, c_y, c_x] += np.sum(this_pixel_value * this_weight)
    return out

# Initialize Hyperparameters
np.random.seed(1)
n_batch = 1
image_height = 4
image_width = 6
channels_in = 5
kernel_size = 3
channels_out = 2

# Define an input image
input_image= np.random.normal(size=(n_batch, channels_in, image_height, image_width))

# Intialize convolution kernel weights
conv_weights = np.random.normal(size=(channels_out, channels_in, kernel_size, kernel_size))

# Perform convolution using PyTorch
conv_results_pytorch = conv_pytorch(input_image, conv_weights, stride=1, pad=1)
print("PyTorch Results")
print(conv_results_pytorch)

# Perform convolution in Numpy
print("Numpy results")
conv_results_numpy = conv_numpy_3(input_image, conv_weights, stride=1, pad=1)
print(conv_results_numpy)

# Perform convolution in numpy (with stride, input, and output layer)
def conv_numpy_4(image, weights, stride=1, pad=1):
    # Perform zero padding
    if pad != 0:
        image = np.pad(image, ((0, 0), (0 ,0), (pad, pad), (pad, pad)),'constant')

    # Determine sizes of image array and kernel weights
    batchSize,  channelsIn, imageHeightIn, imageWidthIn = image.shape
    channelsOut, channelsIn, kernelHeight, kernelWidth = weights.shape

    # Deterime size of output arrays
    imageHeightOut = np.floor(1 + (imageHeightIn - kernelHeight) / stride).astype(int)
    imageWidthOut = np.floor(1 + (imageWidthIn - kernelWidth) / stride).astype(int)

    # Define an array structure for the output of the CNN
    out = np.zeros((batchSize, channelsOut, imageHeightOut, imageWidthOut), dtype=np.float32)

    # For each iamge
    for c_batch in range(batchSize):
      # For the location of each pixel on the final image
      for c_y in range(imageHeightOut):
        for c_x in range(imageWidthOut):
          # For each input and output layer
          for c_channel_out in range(channelsOut):
            for c_channel_in in range(channelsIn):
              # For the location of each pixel on the kernel
              for c_kernel_y in range(kernelHeight):
                for c_kernel_x in range(kernelWidth):
                    # Compute the value of the pixel and the value of the corresopnding kernel weights for each input and output layer
                    # Incorporate stride towards the location of the pixel
                    this_pixel_value = image[c_batch, c_channel_in, c_kernel_y+c_y*stride, c_kernel_x+c_x*stride]
                    this_weight = weights[c_channel_out, c_channel_in, c_kernel_y, c_kernel_x]

                    # Multiply the value of the kernel by the value at the location of the pixel to compute the activation map
                    out[c_batch, c_channel_out, c_y, c_x] += np.sum(this_pixel_value * this_weight)
    return out

# Initialize Parameters
np.random.seed(1)
n_batch = 2
image_height = 4
image_width = 6
channels_in = 5
kernel_size = 3
channels_out = 2

# Define an input image
input_image= np.random.normal(size=(n_batch, channels_in, image_height, image_width))

# Initialize convolution kernel weights
conv_weights = np.random.normal(size=(channels_out, channels_in, kernel_size, kernel_size))

# Perform convolution using PyTorch
conv_results_pytorch = conv_pytorch(input_image, conv_weights, stride=1, pad=1)
print("PyTorch Results")
print(conv_results_pytorch)

# Perform convolution in Numpy
print("Numpy results")
conv_results_numpy = conv_numpy_4(input_image, conv_weights, stride=1, pad=1)
print(conv_results_numpy)
     