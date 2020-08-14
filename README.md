# Pyvlova

A Simple Polyhedral Compiler for NN


# Requirements

- `tvm`
- `numpy`, `astor`, `sympy`
- `isl` ([ChieloNewctle/isl](https://github.com/ChieloNewctle/isl))


# ISL

Please use ([ChieloNewctle/isl](https://github.com/ChieloNewctle/isl)).

After build and install isl, you can use `interface/isl.py` in isl repository
as the dependency for pyvlova.

## Requirements

- `llvm` and `clang`
- `libgmp-dev`
- `libclang-*-dev`, such as `libclang-10-dev`, **IMPORTANT for building the python interface**
- `automake`, `autoconf`, `libtool`
- `pkg-config`

## Build ISL

```bash
git submodule update --init --recursive --progress
autoreconf -i
./configure --with-clang-prefix=/usr/lib/llvm-10    # or your llvm prefix
make -j4
sudo make install
sudo ldconfig
make interface/isl.py
```

# Run tests

To test operators:
```bash
python -m unittest test.test_op
```

