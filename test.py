import numpy as np
x = np.array([[1.0, 0.0, -1.0, -0.0],
              [-1.0, 1.0, -0.0, -0.0]]).astype(np.float16)

print(x)
x = x.astype(np.int8)
assert(np.prod(x.shape) % 4 == 0)

# print(np.prod(x.shape))
x_num = np.prod(x.shape)
x = np.reshape(x, x_num)

group_num = x_num // 4
vec = []
for group in range(group_num):
    temp = np.array(0).astype(np.int8)
    for num in range(4):
        if (x[group * 4 + num] == 1):
            temp = np.left_shift(temp, 1)
            temp = np.bitwise_or(temp, 0)
            temp = np.left_shift(temp, 1)
            temp = np.bitwise_or(temp, 1)
        elif (x[group * 4 + num] == -1):
            temp = np.left_shift(temp, 1)
            temp = np.bitwise_or(temp, 1)
            temp = np.left_shift(temp, 1)
            temp = np.bitwise_or(temp, 1)
        else :
            temp = np.left_shift(temp, 2)
        # print(temp)
        # 
        # 
    vec.append(temp)
res = np.array(vec).astype(np.uint8)
print(res)