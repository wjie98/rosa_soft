import torch

from rosa_soft import RosaContext, rosa_soft_ops, rosa_sufa_ops, rosa_scan_ops


def samx_qkv_slow(qqq, kkk, vvv): # slow, only for reference
    """from https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v8/251024_rosaQKV_run.py
    """
    n=len(qqq); y=[-1]*n; s=2*n+1; t=[None]*s; f=[-1]*s; m=[0]*s; r=[-1]*s; t[0]={}; g=0; u=1; w=h=0; assert n==len(kkk)==len(vvv)
    for i,(q,k) in enumerate(zip(qqq,kkk)):
        p,x=w,h
        while p!=-1 and q not in t[p]: x=m[p] if x>m[p] else x; p=f[p]
        p,x=(t[p][q],x+1) if p!=-1 else (0,0); v=p
        while f[v]!=-1 and m[f[v]]>=x: v=f[v]
        while v!=-1 and (m[v]<=0 or r[v]<0): v=f[v]
        y[i]=vvv[r[v]+1] if v!=-1 else -1; w,h=p,x; j=u; u+=1; t[j]={}; m[j]=m[g]+1; p=g
        while p!=-1 and k not in t[p]: t[p][k]=j; p=f[p]
        if p==-1: f[j]=0
        else:
            d=t[p][k]
            if m[p]+1==m[d]: f[j]=d
            else:
                b=u; u+=1; t[b]=t[d].copy(); m[b]=m[p]+1; f[b]=f[d]; r[b]=r[d]; f[d]=f[j]=b
                while p!=-1 and t[p][k]==d: t[p][k]=b; p=f[p]
        v=g=j
        while v!=-1 and r[v]<i: r[v]=i; v=f[v]
    return [max(0,y) for y in y] # use "0" for both "no-match" and matched "0"


if __name__ == "__main__":
    B, T, H, C, V = 4, 8, 2, 4, 5

    try:    
        for _ in range(10):
            q = torch.randint(0, 2, size=(16,)).tolist()
            k = torch.randint(0, 2, size=(16,)).tolist()
            v = torch.randint(0, 2, size=(16,)).tolist()

            o1 = torch.tensor(samx_qkv_slow(q, k, v))

            query = (torch.tensor([q]).view(1, 1, -1, 1) >> torch.arange(4)) & 1
            key   = (torch.tensor([k]).view(1, 1, -1, 1) >> torch.arange(4)) & 1
            value = (torch.tensor([v]).view(1, 1, -1, 1) >> torch.arange(4)) & 1

            query = query.float()
            key   = key.float()
            value = value.float()
            # print(query.size(), key.size(), value.size())

            o2, _ = RosaContext().update(query, key, value)
            o2 = ((o2 > 0) << torch.arange(4)).sum(dim=-1).squeeze()
            
            o3 = rosa_scan_ops(query, key, value)
            o3 = ((o3 > 0) << torch.arange(4)).sum(dim=-1).squeeze()

            # o4 = RosaContext().update(query, key, value, 0, thresh=-0.1)
            # o4 = ((o4 > 0) << torch.arange(4)).sum(dim=-1).squeeze()

            print(o1)
            print(o2)
            print(o3)
            # print(o4)
            print()
            
            assert (o1 == o2).all()
            assert (o1 == o3).all()
            # assert (o1 == o4).all()

        print("✅ Forward Pass Passed!")
    except AssertionError as e:
        print("❌ Forward Pass Failed!")
        print(e)
    print()


    try:    
        for _ in range(10):
            q = torch.randint(0, 2, size=(8, 2)).float().cuda().view(1, 1, -1, 2).requires_grad_()
            k = torch.randint(0, 2, size=(8, 2)).float().cuda().view(1, 1, -1, 2).requires_grad_()
            v = torch.randint(0, 2, size=(8, 2)).float().cuda().view(1, 1, -1, 2).requires_grad_()

            o = rosa_scan_ops(q, k, v)
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
