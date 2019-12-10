import sympy
import pyvlova.trace
from pyvlova.tensor.dummy_tensor import DummyTensor
import pyvlova.codegen.sympy2isl as s2i
from pyvlova.codegen.loopy2cxx import dump_src_and_header


@pyvlova.trace.TracableCalcFunc
def f(A, n):
    for i in range(n):
        for j in range(3):
            print(s2i.constraints_to_isl_repr(pyvlova.trace.current_status.constraints))
            print(i)
    for i in range(1, n, 2):
        A[i] = i * 2 - 1
        print(s2i.constraints_to_isl_repr(pyvlova.trace.current_status.constraints))
        print(i)
    for i in range(0, n, 2):
        A[i] = i * 2 + 1
        print(i)


@pyvlova.trace.TracableCalcFunc
def mmul(A, B, C):
    assert A.shape[1] == B.shape[0]
    n, c = A.shape
    m = B.shape[1]
    assert C.shape[0] == n and C.shape[1] == m
    for i in range(n):
        for j in range(m):
            C[i][j] = 0
            for k in range(c):
                C[i][j] = C[i][j] + A[i][k] * B[k][j]


def print_kernel_code():
    for i, kernel in enumerate(pyvlova.trace.current_status.loopy_kernels):
        print('=' * 20)
        print(i)
        print('-' * 20)
        src, _ = dump_src_and_header(kernel)
        print(src)
        print('-' * 20)
        print()


if __name__ == '__main__':
    t = DummyTensor(sympy.var('A'), [sympy.var('n')])

    f.trace(t, 8)
    print_kernel_code()

    f.trace(t, sympy.var('n'))
    print_kernel_code()

    A = DummyTensor(sympy.var('A'), [sympy.var('n'), sympy.var('c')])
    B = DummyTensor(sympy.var('B'), [sympy.var('c'), sympy.var('m')])
    C = DummyTensor(sympy.var('C'), [sympy.var('n'), sympy.var('m')])
    mmul.trace(A, B, C)
    print_kernel_code()

    A = DummyTensor(sympy.var('A'), [1024, sympy.var('c')])
    B = DummyTensor(sympy.var('B'), [sympy.var('c'), 2])
    C = DummyTensor(sympy.var('C'), [1024, 2])
    mmul.trace(A, B, C)
    print_kernel_code()
