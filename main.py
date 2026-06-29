import numpy as np
print(np.__version__)

my_list = np.array([1, 20, 3, 4], dtype=np.str_)
print(my_list)

# my_list = my_list * 2

print(my_list)
# print(my_list.sum())
# print(my_list.min())
# print(my_list.max())
print(my_list.dtype)
print(f"{my_list.nbytes} bytes")