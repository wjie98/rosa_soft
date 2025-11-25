import torch
import torch.nn.functional as F

import math
import torch._dynamo as dynamo

from pathlib import Path
from functools import lru_cache

from torch import Tensor
from typing import Dict, List, Tuple, Optional, Union, Literal, cast

import logging
logger = logging.getLogger(__name__)


__all__ = [
    "RapidOnlineSuffixAutomaton",
]

class RosaGSSNode:
    __slots__ = ("occurrences", "length", "suffix_link", "transitions")

    def __init__(self):
        self.occurrences: List[int] = []
        self.length = 0
        self.suffix_link: Optional[RosaGSSNode] = None
        self.transitions: Dict[int, RosaGSSNode] = {}


# class RapidOnlineSuffixAutomatonGraphStructuralSampling:
class RosaGSS:
    def __init__(self):
        self.query_states: List[int] = []
        self.key_states: List[int] = []
        self.value_states: List[int] = []

        self._root: RosaGSSNode = RosaGSSNode()
        self._last_query: RosaGSSNode = self._root
        self._last_key: RosaGSSNode = self._root
    
    def append(
            self,
            query: int,
            key: int,
            value: int,
            default: int = -1,
    ) -> int:

        i = len(self.value_states)

        self.query_states.append(query)
        self.key_states.append(key)
        self.value_states.append(value)

        r = RosaGSSNode()
        r.length = self._last_key.length + 1

        p = self._last_key
        while (p is not None) and (key not in p.transitions):
            p.transitions[key] = r
            p = p.suffix_link
        
        if p is None:
            r.suffix_link = self._root
        else:
            q = p.transitions[key]

            if p.length + 1 == q.length:
                r.suffix_link = q
            else:
                u = RosaGSSNode()
                u.length = p.length + 1
                u.suffix_link = q.suffix_link
                u.occurrences.extend(q.occurrences)
                u.transitions.update(q.transitions)

                q.suffix_link = u
                r.suffix_link = u

                while (p is not None) and (p.transitions.get(key) is q):
                    p.transitions[key] = u
                    p = p.suffix_link
        
        self._last_key = r
        while (r is not None) and (not r.occurrences or r.occurrences[-1] < i):
            r.occurrences.append(i)
            r = r.suffix_link

        j = -1
        
        p = self._last_query
        while (p is not None) and (query not in p.transitions):
            p = p.suffix_link
        
        if p is None:
            self._last_query = self._root
        else:
            self._last_query = p.transitions[query]

            p = self._last_query
            while p is not None:
                if p.length > 0 and p.occurrences and p.occurrences[-1] < i:
                    j = p.occurrences[-1] + 1
                    break
                elif p.length > 0 and len(p.occurrences) > 1 and p.occurrences[-2] < i:
                    j = p.occurrences[-2] + 1
                    break
                p = p.suffix_link
        
        return self.value_states[j] if j >= 0 else default
    
    def extend(
            self,
            query_states: List[int],
            key_states: List[int],
            value_states: List[int],
            default: int = -1,
    ) -> List[int]:
        outs = []
        for q, k, v in zip(query_states, key_states, value_states):
            x = self.append(q, k, v, default)
            outs.append(x)
        return outs


if __name__ == "__main__":
    B, T, H, C, V = 4, 8, 2, 4, 5

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
    
    try:    
        for _ in range(10):
            q = torch.randint(0, 2, size=(8,)).tolist()
            k = torch.randint(0, 2, size=(8,)).tolist()
            v = torch.randint(0, 2, size=(8,)).tolist()

            r = RosaGSS()

            o1 = torch.tensor(samx_qkv_slow(q, k, v))
            o2 = torch.tensor(r.extend(q, k, v, 0))

            print(q)
            print(k)
            print(v)
            print(o1.tolist())
            print(o2.tolist())
            print()

            assert (o1 == o2).all()

        print("✅ Forward Pass Passed!")
    except AssertionError as e:
        print("❌ Forward Pass Failed!")
        print(e)
    print()