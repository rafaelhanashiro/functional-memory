"""Inspect gradient errors from attention tensor-product precision on an H100."""
import importlib.util
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
spec = importlib.util.spec_from_file_location('learned_tests', 'scaffolds/transformer/tests/test_learned_basis.py')
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
for count in (3, 9):
    t.projection(count)
    for sharing in ('global','head','query'):
        rng=np.random.default_rng(352); shape=(2,137,2,32)
        q,k,v,go=[jnp.asarray(rng.normal(size=shape),jnp.bfloat16) for _ in range(4)]
        gain=jnp.asarray(rng.uniform(.5,1.5,shape[-2:]),jnp.float32)
        cshape={'global':(count,), 'head':(2,count),'query':(*shape[:3],count)}[sharing]
        c=jnp.asarray(rng.normal(size=cshape),jnp.float32)
        def joint(fn,*args):
            y,pullback=jax.vjp(fn,*args)
            return y,pullback(go.astype(y.dtype))
        expected=jax.jit(partial(joint,partial(t.dense,count=count)))(q,k,v,gain,c)
        for precision in ('bf16','tf32x3'):
            fn=lambda q,k,v,g,c: t.scalar_attention(q,k,v,kernel=t.basis_kernel(t.projection(count)),basis_coefficients=c,output_norm_scale=g,precision=precision)
            actual=jax.jit(partial(joint,fn))(q,k,v,gain,c)
            errors={}
            for label,a,b in zip(('out','dq','dk','dv','gain','c'),jax.tree.leaves(actual),jax.tree.leaves(expected),strict=True):
                a,b=np.asarray(a,np.float64),np.asarray(b,np.float64)
                errors[label]={'relative':float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-8)),'error_norm':float(np.linalg.norm(a-b)),'norm':float(np.linalg.norm(b))}
                if label=='c':errors[label]['per_basis']=[float(np.linalg.norm(a[...,i]-b[...,i])/max(np.linalg.norm(b[...,i]),1e-8)) for i in range(count)]
            print(count,sharing,precision,errors,flush=True)
