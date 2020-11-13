import torch

torch.manual_seed(432)

x = torch.randn(2, 2).requires_grad_()
gt = torch.ones_like(x) * 10 - 1.0

u = x ** 2
v = u + 9
y = v / 3

loss_fn = torch.nn.MSELoss()
loss = loss_fn(y, gt)

print(f"loss: {loss.item()}")

loss.backward(retain_graph=True)
print(x.grad)

dloss_dx = torch.autograd.grad(outputs=loss, inputs=x)
print(dloss_dx[0])
