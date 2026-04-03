import torch

from rosa_soft import rosa_soft_ops
from rosa_soft import rosa_sufa_ops
from rosa_soft import rosa_scan_ops


if __name__ == "__main__":
    B, T, H, C, V = 4, 8, 2, 4, 5

    try:    
        for _ in range(10):
            for ops in [rosa_soft_ops, rosa_sufa_ops, rosa_scan_ops]:
                q = torch.randint(0, 2, size=(8, 2)).float().cuda().view(1, -1, 1, 2).requires_grad_()
                k = torch.randint(0, 2, size=(8, 2)).float().cuda().view(1, -1, 1, 2).requires_grad_()
                v = torch.randint(0, 2, size=(8, 2)).float().cuda().view(1, -1, 1, 2).requires_grad_()

                o = ops(q, k, v)
                o.sum().backward()

                # print(q.grad.size())
                # print(k.grad.size())
                # print(v.grad.size())

                assert not q.grad.isnan().any()
                assert not k.grad.isnan().any()
                assert not v.grad.isnan().any()

                assert not q.grad.isinf().any()
                assert not k.grad.isinf().any()
                assert not v.grad.isinf().any()

        print("✅ Backward Pass Passed!")
    except AssertionError as e:
        print("❌ Backward Pass Failed!")
        print(e)
    print()
