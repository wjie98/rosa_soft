import torch

from rosa_cpp import RosaContext


if __name__ == "__main__":
    query = torch.randn(3, 8, 100, 8).cuda()
    key = torch.randn(3, 2, 100, 8).cuda()
    value = torch.randn(3, 2, 100, 64).cuda()

    rosa = RosaContext()
    output = rosa.update(query, key, value)
    print(output.shape)
    