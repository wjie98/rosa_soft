import itertools

import numpy as np
import pytest
import torch

from rosa_soft import rosa_bitflip, rosa_hard

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def dp(q, v):
    """Literal longest-suffix DP; no mismatch certificates or owner summaries."""
    b, t, h, _ = q.shape
    out = np.zeros((b, t, h, v.shape[-1]), dtype=np.float64)
    for batch in range(b):
        for head in range(h):
            state = np.zeros(t, dtype=np.int64)
            for i in range(t):
                current = np.zeros(t, dtype=np.int64)
                for j in range(i):
                    if np.array_equal(q[batch, i, head] > 0, q[batch, j, head] > 0):
                        current[j] = 1 + (state[j-1] if j else 0)
                if current.max(initial=0):
                    j = int(np.flatnonzero(current == current.max())[-1])
                    out[batch, i, head] = np.where(v[batch, j+1, head//(h//v.shape[2])] > 0, 1., -1.)
                state = current
    return out


def reference(x, v, dy, tied):
    x, v, dy = [z.detach().cpu().double().numpy() for z in (x, v, dy)]
    base = dp(x, v)
    grads = []
    for side, original in enumerate((x,) if tied == "qkv" else (x, v)):
        grad = np.zeros_like(original)
        for index in np.ndindex(original.shape):
            edited = original.copy()
            edited[index] = -1 if original[index] > 0 else 1
            q = edited if side == 0 else x
            value = edited if side == 1 or tied == "qkv" else v
            delta = ((dp(q, value)-base)*dy).sum()
            grad[index] = delta*(-.5 if original[index]>0 else .5)/(1+abs(original[index]))**2
        grads.append(torch.from_numpy(grad))
    return torch.from_numpy(base), grads


def inputs(tied, dtype=torch.float32, t=7, d=3, dv=5, mask=3):
    torch.manual_seed(599)
    x = torch.randn(1,t,2,d,device="cuda",dtype=dtype)
    if t:
        x[:,0] = 0
    x.requires_grad_(bool(mask&1))
    v = x if tied == "qkv" else torch.randn(1,t,1,dv,device="cuda",dtype=dtype).requires_grad_(bool(mask&2))
    return x,v


@pytest.mark.parametrize("tied,mask", [("qk",1),("qk",2),("qk",3),("qkv",1)])
@pytest.mark.parametrize("dtype", [torch.float16,torch.bfloat16,torch.float32])
@pytest.mark.parametrize("t", [0,1,7])
def test_definition(tied,mask,dtype,t):
    x,v=inputs(tied,dtype,t=t,mask=mask)
    dy=torch.randn(1,t,2,v.size(-1),device="cuda",dtype=dtype)
    expected, grads=reference(x,v,dy,tied)
    y=rosa_bitflip(x,x,v,rows=3,tied=tied)
    assert torch.equal(y.cpu().double(),expected)
    xs=(x,) if tied=="qkv" else (x,v)
    actual=torch.autograd.grad(y,[z for z in xs if z.requires_grad],dy)
    for got,want in zip(actual,[g for z,g in zip(xs,grads) if z.requires_grad]):
        torch.testing.assert_close(got.cpu().float(),want.to(dtype).float(),atol=.008 if dtype==torch.bfloat16 else .001 if dtype==torch.float16 else 3e-5,rtol=.008 if dtype==torch.bfloat16 else .001 if dtype==torch.float16 else 3e-5)


@pytest.mark.parametrize("tied", ["qk","qkv"])
def test_exhaustive_and_high_bit(tied):
    sequences=list(itertools.product((-1.,1.),repeat=5))
    x=torch.tensor(sequences,device="cuda").reshape(-1,5,1,1).requires_grad_()
    v=x if tied=="qkv" else torch.randn_like(x).requires_grad_()
    for x,v in ((x,v),inputs(tied,t=9,d=32,dv=33)):
        dy=torch.randint(-2,3,(x.size(0),x.size(1),x.size(2),v.size(-1)),device="cuda").float()
        expected,grads=reference(x,v,dy,tied)
        y=rosa_bitflip(x,x,v,rows=1,tied=tied)
        assert torch.equal(y.cpu().double(),expected)
        got=torch.autograd.grad(y,(x,) if tied=="qkv" else (x,v),dy)
        for a,b in zip(got,grads):
            torch.testing.assert_close(a.cpu().double(),b,atol=3e-5,rtol=3e-5)


@pytest.mark.parametrize("tied", ["qk", "qkv"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_strided_upstream(tied, dtype):
    torch.manual_seed(937)
    x = torch.randn(2, 9, 3, 3, device="cuda", dtype=dtype, requires_grad=True)
    v = x if tied == "qkv" else torch.randn(2, 9, 1, 5, device="cuda", dtype=dtype, requires_grad=True)
    dy = torch.randn(2, 9, 3, v.size(-1)*2, device="cuda", dtype=dtype)[..., ::2]
    assert not dy.is_contiguous()
    expected, grads = reference(x, v, dy, tied)
    y = rosa_bitflip(x, x, v, rows=3, tied=tied)
    assert torch.equal(y.cpu().double(), expected)
    actual = torch.autograd.grad(y, (x,) if tied == "qkv" else (x, v), dy)
    tolerance = .008 if dtype == torch.bfloat16 else .001 if dtype == torch.float16 else 3e-5
    for a, b in zip(actual, grads):
        torch.testing.assert_close(a.cpu().float(), b.to(dtype).float(), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("tied", ["qk","qkv"])
def test_layout_linearity_and_unlimited(tied):
    x,v=inputs(tied,t=129,d=2)
    with torch.no_grad(): x[:,:65]=1
    x=x.transpose(1,2).contiguous().transpose(1,2).detach().requires_grad_()
    v=x if tied=="qkv" else v
    a,b=[torch.randn(1,129,2,v.size(-1),device="cuda") for _ in range(2)]
    xs=(x,) if tied=="qkv" else (x,v)
    all_grads=[]
    for rows,dy in ((1,a),(31,b),(128,2*a-3*b)):
        y=rosa_bitflip(x,x,v,rows=rows,tied=tied)
        assert torch.equal(y,rosa_hard(x,x,v)[0])
        all_grads.append(torch.autograd.grad(y,xs,dy))
    for u,v,w in zip(*all_grads):
        torch.testing.assert_close(2*u-3*v,w,atol=1e-4,rtol=1e-4)


@pytest.mark.parametrize("tied,mask", [("qk",1),("qk",2),("qk",3),("qkv",1)])
def test_compile_and_opcheck(tied,mask):
    torch._dynamo.reset()
    x,v=inputs(tied,mask=mask)
    torch.library.opcheck(torch.ops.rosa_soft.joint_forward.default,(x,v,3,tied=="qkv"))
    fn=torch.compile(rosa_bitflip,fullgraph=True,dynamic=True)
    for t in (7,1,0):
        x,v=inputs(tied,t=t,mask=mask)
        dy=torch.ones(1,t,2,v.size(-1),device="cuda")
        y=fn(x,x,v,rows=3,tied=tied)
        expected,grads=reference(x,v,dy,tied)
        assert torch.equal(y.cpu().double(),expected)
        xs=(x,) if tied=="qkv" else (x,v)
        got=torch.autograd.grad(y,[z for z in xs if z.requires_grad],dy)
        for a,b in zip(got,[g for z,g in zip(xs,grads) if z.requires_grad]):
            torch.testing.assert_close(a.cpu().double(),b,atol=3e-5,rtol=3e-5)


def test_binding_is_explicit_and_validation():
    x,v=inputs("qkv",t=5)
    dy=torch.randn_like(x)
    independent=torch.autograd.grad(rosa_bitflip(x,x,x),x,dy)[0]
    joint=torch.autograd.grad(rosa_bitflip(x,x,x,tied="qkv"),x,dy)[0]
    assert not torch.allclose(independent,joint)
    for tied,args in (("qk",(x,x.clone(),x)),("qkv",(x,x,x.clone())),("auto",(x,x,x))):
        with pytest.raises(ValueError,match="tied"):
            rosa_bitflip(*args,tied=tied)
    with pytest.raises(RuntimeError,match="QKV requires"):
        torch.ops.rosa_soft.joint_forward(x,x.half(),3,True)
    with pytest.raises(RuntimeError,match="invalid joint"):
        torch.ops.rosa_soft.joint_backward(x,x,dy,torch.empty(0,device="cuda",dtype=torch.int32),
                                          torch.empty(0,device="cuda",dtype=torch.int64),3,True,1)
    before=torch.are_deterministic_algorithms_enabled()
    warn=torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(RuntimeError,match="deterministic"):
            rosa_bitflip(x,x,x,tied="qkv").sum().backward()
    finally:
        torch.use_deterministic_algorithms(before,warn_only=warn)


@pytest.mark.parametrize("tied", ["qk","qkv"])
def test_changed_graph_on_nondefault_stream(tied):
    x,v=inputs(tied,t=9)
    xs=(x,) if tied=="qkv" else (x,v)
    dy=torch.randn(1,9,2,v.size(-1),device="cuda")
    def step():
        y=rosa_bitflip(x,x,v,rows=3,tied=tied)
        return y,torch.autograd.grad(y,xs,dy)
    stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step(); stream.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream): y,grads=step()
        for _ in range(3):
            with torch.no_grad():
                for z in xs: z.copy_(torch.randn_like(z))
                dy.copy_(torch.randn_like(dy))
            graph.replay()
            out,want=reference(x,v,dy,tied)
            assert torch.equal(y.cpu().double(),out)
            for a,b in zip(grads,want):
                torch.testing.assert_close(a.cpu().double(),b,atol=3e-5,rtol=3e-5)
    torch.cuda.current_stream().wait_stream(stream)


class Reference(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,v,tied):
        ctx.save_for_backward(x,v); ctx.tied=tied
        return torch.from_numpy(dp(x.detach().cpu().numpy(),v.detach().cpu().numpy())).to(x)

    @staticmethod
    def backward(ctx,dy):
        x,v=ctx.saved_tensors
        _,grads=reference(x,v,dy,ctx.tied)
        return grads[0].to(x),grads[1].to(v) if ctx.tied=="qk" else None,None


@pytest.mark.parametrize("tied", ["qk","qkv"])
@pytest.mark.parametrize("compiled", [False,True])
def test_checkpoint_model_gradients(tied,compiled):
    from torch import nn
    from torch.utils.checkpoint import checkpoint
    torch._dynamo.reset()
    torch.manual_seed(883)
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.input=nn.Linear(5,16)
            self.proj=nn.ModuleList([nn.Linear(16,2 if tied=="qkv" else 5) for _ in range(2)])
            self.fuse=nn.ModuleList([nn.Sequential(nn.Linear(18 if tied=="qkv" else 19,16),nn.SiLU()) for _ in range(2)])
            self.out=nn.Linear(16,7)
        def forward(self,x,oracle=False):
            h=self.input(x)
            for proj,fuse in zip(self.proj,self.fuse):
                def layer(h,proj=proj,fuse=fuse):
                    z=proj(h)
                    q=z[...,:2].unsqueeze(2)
                    v=q if tied=="qkv" else z[...,2:].unsqueeze(2)
                    y=Reference.apply(q,v,tied) if oracle else rosa_bitflip(q,q,v,tied=tied,rows=3)
                    return h+fuse(torch.cat((h,y.flatten(-2)),-1))
                h=checkpoint(layer,h,use_reentrant=False)
            return self.out(h)
    model=Model().cuda(); params=tuple(model.parameters())
    call=torch.compile(model,fullgraph=True) if compiled else model
    data=torch.randn(1,5,5,device="cuda")
    labels=torch.randint(7,(1,5),device="cuda")
    opt=torch.optim.AdamW(params,lr=.001)
    for _ in range(3):
        gradients=[]
        for fn in (lambda:model(data,True),lambda:call(data)):
            loss=torch.nn.functional.cross_entropy(fn().flatten(0,1),labels.flatten())
            gradients.append(torch.autograd.grad(loss,params))
        for a,b in zip(*gradients): torch.testing.assert_close(a,b,atol=3e-5,rtol=3e-5)
        opt.zero_grad(set_to_none=True)
        for p,g in zip(params,gradients[1]): p.grad=g
        opt.step()
