project('flax-attention', 'cuda')
cuda_args = [
                        '-gencode',
                        'arch=compute_80,code=sm_80',
                        '-w',
                        '-O3',
                        '-std=c++17',
                        '-U__CUDA_NO_HALF_OPERATORS__',
                        '-U__CUDA_NO_HALF_CONVERSIONS__',
                        '-U__CUDA_NO_HALF2_OPERATORS__',
                        '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                        '--expt-relaxed-constexpr',
                        '--expt-extended-lambda',
                        '--use_fast_math',
                    ]
add_global_arguments(cuda_args, language : 'cuda')
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

