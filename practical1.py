from torch import Tensor
import torch as torch
import time as time

# 1 - Multiple views of a storage
def multiple_views():
    x = Tensor(13, 13)
    x.fill_(1)
    x.narrow(0, 1, 1).fill_(2)
    x.narrow(0, 6, 1).fill_(2)
    x.narrow(0, 11, 1).fill_(2)
    x.narrow(1, 1, 1).fill_(2)
    x.narrow(1, 6, 1).fill_(2)
    x.narrow(1, 11, 1).fill_(2)
    z = x.narrow(0, 3, 2).narrow(1, 3, 2)
    z.fill_(3)
    z = x.narrow(0, 8, 2).narrow(1, 3, 2)
    z.fill_(3)
    z = x.narrow(0, 3, 2).narrow(1, 8, 2)
    z.fill_(3)
    z = x.narrow(0, 8, 2).narrow(1, 8, 2)
    z.fill_(3)
    return x

# 2 - Eigendecomposition
def eigen_decomp():
    x = Tensor(20, 20).normal_()
    D = torch.diag(torch.arange(1, 21))
    y = Tensor.inverse(x)
    z = torch.mm(y, torch.mm(D, x))
    return Tensor.eig(z)


# 3 - Flops per second
def flops_per_second():
    x = Tensor(5000, 5000).normal_()
    y = Tensor(5000, 5000).normal_()
    start_time = time.perf_counter()
    torch.mm(x, y)
    end_time = time.perf_counter()
    fps = 5000*5000*5000 / (end_time - start_time) # matrix multiplication is approx. O(n^3)
    return fps

# 4 - Playing with strides
def mul_row(x):
    start_time = time.perf_counter()
    nrows = (x.size())[0]
    for i in range(nrows):
        y = x.narrow(0, i, 1)
        y *= (i+1)
    end_time = time.perf_counter()
    return end_time - start_time

def mul_row_fast(x):
    start_time = time.perf_counter()
    nrows = (x.size())[0]
    y = torch.arange(1, nrows+1).expand(x.size()[1], nrows).t()
    end_time = time.perf_counter()
    return end_time - start_time

def part_four():
    m = Tensor(10000, 400).fill_(2.0)
    t1 = mul_row(m)
    t2 = mul_row_fast(m)
    return t1 / t2

print("Part one matrix: \n", multiple_views())
print("Part two eigendecomposition: \n", eigen_decomp())
print("Part three billion flops per second: %d\n" % (flops_per_second() / 10**9))
print("Part four time(mul_row) / time(mul_row_fast): %d\n" % part_four())