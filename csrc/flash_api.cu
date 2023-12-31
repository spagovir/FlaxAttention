#include "flash.h"
#include <pybind11/pybind11.h>
int foo()
{
    return 42;
}

PYBIND11_MODULE(flax_attn, m)
{
    m.def("foo", &foo, "");
}
