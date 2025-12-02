import torch

from rosa_cpp.layers import RosaAttention


if __name__ == "__main__":
    B, T, C = 3, 100, 64

    class Config:
        def __init__(self):
            self.hidden_size = C
            self.rosa_proxy_type = "gss"

    layer = RosaAttention(Config(), 0)
    layer.train()

    try:    
        for i in range(10):
            x = torch.randn(B, T, C).requires_grad_()
            
            layer(x).sum().backward()

            assert not x.grad.isnan().any()
            assert not x.grad.isinf().any()

            print(f"{i:02d}: {x.grad.size()}")

            for name, p in layer.named_parameters():
                assert not p.grad.isnan().any()
                assert not p.grad.isinf().any()

                print(f"  {name}: {p.grad.size()}")

        print("✅ Backward Pass Passed!")
    except AssertionError as e:
        print("❌ Backward Pass Failed!")
        print(e)
    print()
