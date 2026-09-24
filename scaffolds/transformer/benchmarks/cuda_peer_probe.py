"""Temporary direct CUDA copy check outside JAX and the model."""
import ctypes as c
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

lib=Path(sys.prefix)/'lib/python3.11/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12'
cuda=c.CDLL(str(lib))
signatures={
 'cudaGetDeviceCount':[c.POINTER(c.c_int)], 'cudaSetDevice':[c.c_int],
 'cudaMalloc':[c.POINTER(c.c_void_p),c.c_size_t], 'cudaFree':[c.c_void_p],
 'cudaMemcpy':[c.c_void_p,c.c_void_p,c.c_size_t,c.c_int],
 'cudaMemcpyPeer':[c.c_void_p,c.c_int,c.c_void_p,c.c_int,c.c_size_t],
 'cudaMemset':[c.c_void_p,c.c_int,c.c_size_t], 'cudaDeviceSynchronize':[],
 'cudaDeviceCanAccessPeer':[c.POINTER(c.c_int),c.c_int,c.c_int],
 'cudaDeviceEnablePeerAccess':[c.c_int,c.c_uint],
}
for name,types in signatures.items():
 getattr(cuda,name).argtypes=types
 getattr(cuda,name).restype=c.c_int
def call(name,*args):
 status=getattr(cuda,name)(*args)
 if status: raise RuntimeError(f'{name} returned {status}')
count=c.c_int();call('cudaGetDeviceCount',c.byref(count));assert count.value==4
output=Path(sys.argv[1]);results={'library':str(lib),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'peer_capability':[],'copies':[]}
def save(): output.write_text(json.dumps(results,indent=2)+'\n')
for dst in range(4):
 for src in range(4):
  if src==dst:continue
  can=c.c_int();call('cudaDeviceCanAccessPeer',c.byref(can),dst,src)
  results['peer_capability'].append({'source':src,'destination':dst,'supported':bool(can.value)})
save()
for enabled in (False,True):
 if enabled:
  for pair in results['peer_capability']:
   if pair['supported']:
    call('cudaSetDevice',pair['destination']);call('cudaDeviceEnablePeerAccess',pair['source'],0)
 for size in (4096,1048576,16777216):
  host=np.arange(size//4,dtype=np.uint32)*np.uint32(1664525)+np.uint32(1013904223)
  target=np.empty_like(host); pointers=[]
  for device in range(4):
   call('cudaSetDevice',device);ptr=c.c_void_p();call('cudaMalloc',c.byref(ptr),size);pointers.append(ptr)
  for src in range(4):
   call('cudaSetDevice',src);call('cudaMemcpy',pointers[src],host.ctypes.data,size,1);call('cudaDeviceSynchronize')
   for dst in range(4):
    if src==dst:continue
    call('cudaSetDevice',dst);call('cudaMemset',pointers[dst],0xA5,size);call('cudaDeviceSynchronize')
    call('cudaMemcpyPeer',pointers[dst],dst,pointers[src],src,size);call('cudaDeviceSynchronize')
    call('cudaMemcpy',target.ctypes.data,pointers[dst],size,2)
    mismatches=int(np.count_nonzero(target!=host))
    results['copies'].append({'peer_enabled':enabled,'bytes':size,'source':src,'destination':dst,
      'mismatch_uint32':mismatches,'expected_first':host[:8].tolist(),'actual_first':target[:8].tolist()})
    save()
  for device,ptr in enumerate(pointers):call('cudaSetDevice',device);call('cudaFree',ptr)
results['status']='complete';save()
print(json.dumps({'total':len(results['copies']),'failed':sum(x['mismatch_uint32']>0 for x in results['copies'])}))
