import importlib.util
import traceback

print('flash_attn_spec', importlib.util.find_spec('flash_attn'))
try:
    import flash_attn.cute as cute
    print('flash_attn_cute_import', True)
    print('has_flash_attn_func', hasattr(cute, 'flash_attn_func'))
except Exception as e:
    print('flash_attn_cute_import', False)
    print('flash_attn_cute_error', repr(e))
    traceback.print_exc()
