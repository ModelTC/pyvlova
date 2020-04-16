# Pyvlova

A Simple Polyhedral Compiler for NN


# Requirements

<!--
This project requires `astor`, `sympy`, `numpy`.

And the newest `loopy` ([loopy](https://github.com/inducer/loopy)).
-->

- the newest `tvm`
- `numpy`, `pytorch`, `astor`, `sympy`
- `isl` ([ChieloNewctle/isl](https://github.com/ChieloNewctle/isl))


# ISL

Please use ([ChieloNewctle/isl](https://github.com/ChieloNewctle/isl)).

After build and install isl, you can use `interface/isl.py` in isl repository
as the dependency for pyvlova.

## Requirements

- `llvm` and `clang`
- `libclang-*-dev`, such as `libclang-10-dev`, **IMPORTANT for building interface**
- `automake`, `autoconf`, `libtool`
- `pkg-config`

## Build

```
git submodule update --init --recursive --progress
autoreconf -i
./configure --with-clang-prefix=/usr/lib/llvm-10    # or your llvm prefix
make -j8
make install
make interface/isl.py
```
