from unittest import result

import torch

# z = torch.zeros(5, 3)
# print(z)
# print(z.dtype)
# i = torch.ones((5, 3), dtype=torch.int16)
# print(i)

# A scalar is a single number and in tensor-speak it's a zero dimension tensor.
# Scalar
# scalar = torch.tensor(7)
# print(scalar)

# We can check the dimensions of a tensor using the ndim attribute.
# print(scalar.ndim)

# What if we wanted to retrieve the number from the tensor?
# As in, turn it from torch.Tensor to a Python integer?
# To do we can use the item() method.
# You can tell the number of dimensions a tensor in PyTorch has by the number of square brackets on the outside ([) and you only need to count one side.
# How many square brackets does vector have?
# Another important concept for tensors is their shape attribute. The shape tells you how the elements inside them are arranged.
# Check shape of vector
# vector = torch.tensor([7, 7])
# vector.shape
# The above returns torch.Size([2]) which means our vector has a shape of [2]. This is because of the two elements we placed inside the square brackets ([7, 7]).

# how to create a tensor of random numbers.
# Create a random tensor of size (3, 4)
# random_tensor = torch.rand(size=(3, 4))
# print(random_tensor, random_tensor.dtype)
# isCudaAvail = torch.backends.mps.is_available()
# print(isCudaAvail)

# Set device type
# device = "mps" if torch.backends.mps.is_available() else "cpu"
# print(device)

# Create tensor (default on CPU)
# tensor = torch.tensor([1, 2, 3])

# Tensor not on GPU
# print(tensor, tensor.device)

# Move tensor to GPU (if available)
# tensor_on_gpu = tensor.to(device)
# print(tensor_on_gpu, tensor_on_gpu.device)

# random_tensor = torch.rand(1,7)
# print(random_tensor)
# print("Shape of tensor: ", random_tensor.shape)

# 3. Perform a matrix multiplication on the tensor from 2 with another random tensor with shape (1, 7) (hint: you may have to transpose the second tensor).
# tensor1 = torch.rand(7,7)
# tensor2 = torch.rand(1,7)

# result = torch.matmul(tensor1, tensor2)
# print(result)
# print(result.shape)

# 1. Set the random seed to 0 for reproducibility
# torch.manual_seed(0)

# 2. Create the first random tensor (Shape: 7, 7)
# tensor1 = torch.rand(7, 7)

# 3. Create the second random tensor (Shape: 1, 7)
# tensor2 = torch.rand(1, 7)

# 4. Transpose and perform matrix multiplication
# result = torch.matmul(tensor1, tensor2.T)

# Print results (These will be identical every time you run this)
# print("Tensor 1:\n", tensor1)
# print("\nTensor 2 Transposed:\n", tensor2.T)
# print("\nMultiplication Result:\n", result)

# device = "mps" if torch.backends.mps.is_available() else "cpu"
# print(f"Using device: {device}")
#
# torch.manual_seed(1234)
# if torch.backends.mps.is_available():
#     torch.mps.manual_seed(1234)
# tensor_A = torch.rand(2, 3).to(device)
# tensor_B = torch.rand(2, 3).to(device)
#
# multipliedResult = torch.matmul(tensor_A, tensor_B.t())
# print(multipliedResult)

device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(1234)

if torch.backends.mps.is_available():
    torch.mps.manual_seed(1234)

tenso_1 = torch.randn(2,3).to(device)
tenso_2 = torch.randn(2,3).to(device)
result = torch.matmul(tenso_1, tenso_2.t())
print(result)

print(torch.max(result))
print(torch.min(result))