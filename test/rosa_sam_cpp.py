import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rosa_soft import RosaRuntime


def samx_qkv_slow(qqq, kkk, vvv):
    """Reference from BlinkDL RWKV-v8 ROSA demo, with endpos returned."""
    n = len(qqq)
    y = [0] * n
    pos = [-1] * n
    s = 2 * n + 1
    t = [None] * s
    f = [-1] * s
    m = [0] * s
    r = [-1] * s
    t[0] = {}
    g = 0
    u = 1
    w = h = 0
    assert n == len(kkk) == len(vvv)
    for i, (q, k) in enumerate(zip(qqq, kkk)):
        p, x = w, h
        while p != -1 and q not in t[p]:
            x = m[p] if x > m[p] else x
            p = f[p]
        p, x = (t[p][q], x + 1) if p != -1 else (0, 0)
        v = p
        while f[v] != -1 and m[f[v]] >= x:
            v = f[v]
        while v != -1 and (m[v] <= 0 or r[v] < 0):
            v = f[v]
        pos[i] = r[v] if v != -1 else -1
        y[i] = vvv[pos[i] + 1] if pos[i] >= 0 else 0
        w, h = p, x

        j = u
        u += 1
        t[j] = {}
        m[j] = m[g] + 1
        p = g
        while p != -1 and k not in t[p]:
            t[p][k] = j
            p = f[p]
        if p == -1:
            f[j] = 0
        else:
            d = t[p][k]
            if m[p] + 1 == m[d]:
                f[j] = d
            else:
                b = u
                u += 1
                t[b] = t[d].copy()
                m[b] = m[p] + 1
                f[b] = f[d]
                r[b] = r[d]
                f[d] = f[j] = b
                while p != -1 and t[p][k] == d:
                    t[p][k] = b
                    p = f[p]
        v = g = j
        while v != -1 and r[v] < i:
            r[v] = i
            v = f[v]
    return y, pos


def packed_reference(q, k, v, cu_seqlens=None):
    if cu_seqlens is None:
        B, T, H = q.shape
        Hv = v.size(2)
        group = H // Hv
        out = torch.empty((B, T, H), dtype=torch.uint8)
        endpos = torch.empty((B, T, H), dtype=torch.int64)
        for b in range(B):
            for h in range(H):
                y, p = samx_qkv_slow(
                    q[b, :, h].tolist(),
                    k[b, :, h].tolist(),
                    v[b, :, h // group].tolist(),
                )
                out[b, :, h] = torch.tensor(y, dtype=torch.uint8)
                endpos[b, :, h] = torch.tensor(p, dtype=torch.int64)
        return out, endpos

    total, H = q.shape
    Hv = v.size(1)
    group = H // Hv
    out = torch.empty((total, H), dtype=torch.uint8)
    endpos = torch.empty((total, H), dtype=torch.int64)
    offsets = cu_seqlens.cpu().tolist()
    for b in range(len(offsets) - 1):
        start, stop = offsets[b], offsets[b + 1]
        for h in range(H):
            y, p = samx_qkv_slow(
                q[start:stop, h].tolist(),
                k[start:stop, h].tolist(),
                v[start:stop, h // group].tolist(),
            )
            out[start:stop, h] = torch.tensor(y, dtype=torch.uint8)
            endpos[start:stop, h] = torch.tensor(p, dtype=torch.int64)
    return out, endpos


def unpack_bits(x, bits, device=None):
    if device is None:
        device = x.device
    shifts = torch.arange(bits, device=device, dtype=torch.int16)
    x = x.to(device=device, dtype=torch.int16)
    bits_out = ((x.unsqueeze(-1) >> shifts) & 1).float()
    return torch.where(bits_out != 0, torch.ones_like(bits_out), -torch.ones_like(bits_out))


def pack_bits(x):
    bits = x.size(-1)
    weights = (1 << torch.arange(bits, device=x.device, dtype=torch.int16)).view(
        *([1] * (x.dim() - 1)), bits
    )
    return ((x > 0).to(torch.int16) * weights).sum(dim=-1).to(torch.uint8)


def check_packed_dense():
    torch.manual_seed(0)
    B, T, H, Hv, bits = 3, 32, 4, 2, 4
    q = torch.randint(0, 1 << bits, (B, T, H), dtype=torch.uint8)
    k = torch.randint(0, 1 << bits, (B, T, H), dtype=torch.uint8)
    v = torch.randint(0, 1 << bits, (B, T, Hv), dtype=torch.uint8)
    ref, ref_pos = packed_reference(q, k, v)

    for backend in ("map", "compact"):
        with RosaRuntime(H, Hv, bits, bits, backend=backend) as rt:
            out, pos = rt.update_packed(q, k, v)
            assert torch.equal(out, ref), backend
            assert torch.equal(pos, ref_pos), backend
            states, edges, values = rt.stats()
            assert states > 0 and edges > 0 and values == B * H * T


def check_varlen():
    torch.manual_seed(1)
    lengths = [5, 17, 9]
    H, Hv, bits = 6, 3, 3
    total = sum(lengths)
    cu = torch.tensor([0, 5, 22, 31], dtype=torch.int32)
    q = torch.randint(0, 1 << bits, (total, H), dtype=torch.uint8)
    k = torch.randint(0, 1 << bits, (total, H), dtype=torch.uint8)
    v = torch.randint(0, 1 << bits, (total, Hv), dtype=torch.uint8)
    ref, ref_pos = packed_reference(q, k, v, cu)

    with RosaRuntime(H, Hv, bits, bits, backend="compact") as rt:
        out, pos = rt.update_packed(q, k, v, cu_seqlens=cu)
        assert torch.equal(out, ref)
        assert torch.equal(pos, ref_pos)


def check_unpacked_cuda_if_available():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(2)
    device = "cuda"
    B, T, H, Hv, bits = 2, 24, 4, 2, 4
    q_pack = torch.randint(0, 1 << bits, (B, T, H), device=device, dtype=torch.uint8)
    k_pack = torch.randint(0, 1 << bits, (B, T, H), device=device, dtype=torch.uint8)
    v_pack = torch.randint(0, 1 << bits, (B, T, Hv), device=device, dtype=torch.uint8)
    q = unpack_bits(q_pack, bits)
    k = unpack_bits(k_pack, bits)
    v = unpack_bits(v_pack, bits)
    ref, _ = packed_reference(q_pack.cpu(), k_pack.cpu(), v_pack.cpu())

    with RosaRuntime(H, Hv, bits, bits) as rt:
        out, _ = rt.update(q, k, v, return_packed=False)
        assert torch.equal(pack_bits(out).cpu(), ref)


def check_async_cuda_if_available():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(3)
    device = "cuda"
    B, T, H, Hv, bits = 2, 32, 4, 2, 4
    q = torch.randint(0, 1 << bits, (B, T, H), device=device, dtype=torch.uint8)
    k = torch.randint(0, 1 << bits, (B, T, H), device=device, dtype=torch.uint8)
    v = torch.randint(0, 1 << bits, (B, T, Hv), device=device, dtype=torch.uint8)

    with RosaRuntime(H, Hv, bits, bits) as rt_block:
        out_block, pos_block = rt_block.update_packed(q, k, v)

    stream = torch.cuda.Stream()
    with RosaRuntime(H, Hv, bits, bits) as rt_async:
        work = rt_async.update_packed(q, k, v, stream=stream, async_op=True)
        out_async, pos_async = work.wait()

    assert torch.equal(out_async.cpu(), out_block.cpu())
    assert torch.equal(pos_async.cpu(), pos_block.cpu())


def check_close():
    q = torch.zeros(1, 4, 2, dtype=torch.uint8)
    k = torch.zeros_like(q)
    v = torch.zeros(1, 4, 1, dtype=torch.uint8)

    rt = RosaRuntime(2, 1, 2, 2)
    rt.update_packed(q, k, v)
    assert rt.stats()[2] == 8
    rt.update_packed(q, k, v)
    assert rt.stats()[2] == 16
    rt.close()
    assert rt.stats() == (0, 0, 0)
    try:
        rt.update_packed(q, k, v)
    except RuntimeError:
        return
    raise AssertionError("closed runtime accepted update")


if __name__ == "__main__":
    check_packed_dense()
    check_varlen()
    check_unpacked_cuda_if_available()
    check_async_cuda_if_available()
    check_close()
    print("RosaRuntime tests passed")
