import loopy as lp

def dump_src_and_header(kernel):
    return lp.generate_code(kernel)[0], lp.generate_header(kernel)
