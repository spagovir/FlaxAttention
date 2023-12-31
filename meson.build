project('flax-attention', 'cuda')
py = import('python').find_installation(pure: false)
flash_dir = 'csrc/flash-attention/csrc/flash_attn/src'
flash_cu_srcs = []
bfs = ['bwd','fwd']
hdims = [32,64,96,128,160,192,256]
dtypes = ['fp16', 'bf16']
foreach bf : bfs foreach hdim : hdims foreach dtype : dtypes
    flash_cu_srcs += flash_dir / f'flash_@bf@_hdim@hdim@_@dtype@_sm80.cu'
endforeach endforeach endforeach
# dep = dependency('cuda', version : '>=12.2', modules : ['cublas'])
includes = [
    flash_dir, 
    'csrc/extern/pybind11/include',
    'csrc/LibCandle/src', 
    'csrc/flash-attention/csrc/cutlass/include']
flash_lib = library('flash_lib',
    flash_cu_srcs,
    include_directories : includes,
 #   dependencies : dep
)
py.extension_module(
    'flax_attn', 
    'csrc/flash_api.cu',
    link_with : flash_lib,
    include_directories : includes,
#    dependencies : dep,
    install : true
)

