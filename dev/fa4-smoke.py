import torch
import nanochat.flash_attention as fa

print('HAS_FA3', fa.HAS_FA3)
print('HAS_FA4', getattr(fa, 'HAS_FA4', None))
print('FAST_ATTN_BACKEND', getattr(fa, 'FAST_ATTN_BACKEND', None))
print('USE_FA3', getattr(fa, 'USE_FA3', None))
print('USE_FA4', getattr(fa, 'USE_FA4', None))

q = torch.randn(1, 16, 2, 64, device='cuda', dtype=torch.bfloat16)
k = torch.randn(1, 16, 2, 64, device='cuda', dtype=torch.bfloat16)
v = torch.randn(1, 16, 2, 64, device='cuda', dtype=torch.bfloat16)

y = fa.flash_attn.flash_attn_func(q, k, v, causal=False)
torch.cuda.synchronize()
print('out_shape', tuple(y.shape))
print('out_dtype', y.dtype)
