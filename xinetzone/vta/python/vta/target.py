from tvm.target.target import Target, _merge_opts

def vta_target(model="unknown", options=None):
    opts = ["-device=vta", "-keys=vta,cpu", "-model=%s" % model]
    opts = _merge_opts(opts, options)
    return Target(" ".join(["ext_dev"] + opts))
