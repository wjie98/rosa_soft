import torch

from rosa_soft import RosaContext, RosaCache


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
    B, T, H, C = 32, 1024, 8, 2

    for i in range(10):
        try:
            Q = torch.randint(0, 1 << C, size=(B, T, H)).cuda()
            K = torch.randint(0, 1 << C, size=(B, T, H)).cuda()
            V = torch.randint(0, 1 << C, size=(B, T, H)).cuda()

            ground_truth = []
            for b in range(B):
                for h in range(H):
                    ground_truth.append(samx_qkv_slow(Q[b, :, h].tolist(), K[b, :, h].tolist(), V[b, :, h].tolist()))
            ground_truth = torch.tensor(ground_truth).view(B, H, T).permute(0, 2, 1).cuda()

            o1, p1 = RosaCache(B, H).update(Q, K, V)
            o2 = torch.gather(V, dim=1, index=p1+1) * (p1 >= 0)

            assert (o1 == ground_truth).all()
            assert (o2 == ground_truth).all()
            print(f"✅ Cache Pass Passed! {i}")

            Q = Q.unsqueeze(-1) >> torch.arange(C, device=Q.device)
            K = K.unsqueeze(-1) >> torch.arange(C, device=K.device)
            V = V.unsqueeze(-1) >> torch.arange(C, device=V.device)

            Q = torch.where(Q & 1 > 0, 1.0, -1.0)
            K = torch.where(K & 1 > 0, 1.0, -1.0)
            V = torch.where(V & 1 > 0, 1.0, -1.0)

            o3, *_ = RosaContext(B, H).update(Q, K, V)
            o3 = (o3 > 0).long() << torch.arange(C, device=o3.device)
            o3 = o3.sum(dim=-1)

            assert (o3 == ground_truth).all()
            print(f"✅ Context Pass Passed! {i}")
            
        except AssertionError as e:
            print(f"❌ Cache/Context Pass Failed! {e}")
