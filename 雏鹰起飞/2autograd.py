# 把必要的包给导入
import torch
from torch.autograd import Variable

# 如果我们想计算  f=3×x3+4×x2+6f=3×x3+4×x2+6  的导数，该如何做呢？

def fn(x):
    y = 3 * x.pow(3) + 4 * x.pow(2) + 6
    return y

# 将x设为Variable是为了满足模型训练格式要求
# 如自动求导、nn.Sequential及nn.Linear等神经网络的计算
x1 = Variable(torch.Tensor([1]), requires_grad=True)

y1 = fn(x1)  # fn(x1)的输入x1是一个Variable,输出也会自动以Variable的形式返回。

print(y1)  # print: tensor([13.], grad_fn=<AddBackward0>)，后面那个参数应该是记录生成y1的最后一步操作，即进行加法计算
# 还有就是在fn里面得到的y1必须要有x1的参与才能使得y1也是一个Variable对象

y1.backward() # 自动求导

# 查看x1的梯度（可以print一下)。即fn(x)对变量x在x1处的偏导数
x1.grad # 通过这个写法可以看出y1.backward()的计算结果会存储在x1这个Variable对象的grad属性中


# 通过调用 backward() 函数，我们自动求出了在 x = 1 的时候的导数
# 需要注意的一点是：如果我们输入的 Tensor 不是一个标量，而是矢量（多个值）。
# 那么，我们在调用backward()之前，需要让结果变成标量 才能求出导数。
# 也就是说如果不将 Y 的值变成标量，就会报错。（可以尝试把mean()给取消，看看是不是报错了）

x2 = Variable(torch.Tensor([[1, 2], [3, 4]]), requires_grad=True)

y2 = fn(x2).mean() # 将结果变成标量，这样就不会报错了

y2.backward() # 自动求导

x2.grad # 查看梯度
