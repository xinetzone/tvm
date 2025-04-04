# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=unused-argument
"""tvm.contrib.msc.core.transform.pattern"""

from typing import Mapping, Tuple, Dict, List
from functools import partial

import tvm
from tvm.relax.dpl import pattern as relax_pattern

from tvm.relax.transform import PatternCheckContext
from tvm.relax.backend.pattern_registry import register_patterns
from tvm.contrib.msc.core.utils.namespace import MSCMap, MSCKey
from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.core import _ffi_api


def msc_attrs_getter(
    annotated_expr: Dict[str, tvm.relax.Expr],
    anchor: str = "out",
    output: str = None,
    inputs: List[str] = None,
) -> Dict[str, str]:
    """Get attributes for fused pattern

    Parameters
    ----------
    annotated_expr: dict<str,Expr>
        The annotated exprs during fus pattern
    anchor: str
        The anchor key of expr

    Returns
    -------
    attrs: dict<str,str>
        The extra attributes for msc.
    """

    attrs = {}
    # get name
    fused_cnt = MSCMap.get(MSCKey.FUSED_CNT, 0)
    unique_name = "msc_fused_" + str(fused_cnt)
    if anchor in annotated_expr:
        name = msc_utils.get_expr_name(annotated_expr[anchor])
        if name:
            unique_name = name
    MSCMap.set(MSCKey.FUSED_CNT, fused_cnt + 1)
    attrs[_ffi_api.ToAttrKey("unique")] = unique_name
    # get output layout
    output = output or anchor
    if output in annotated_expr:
        attrs[_ffi_api.ToAttrKey("layout")] = msc_utils.get_expr_layout(annotated_expr[output])
    if inputs:
        layouts = {}
        for i in inputs:
            if i not in annotated_expr:
                continue
            in_name = msc_utils.get_expr_name(annotated_expr[i])
            if not in_name:
                continue
            layouts[in_name] = msc_utils.get_expr_layout(annotated_expr[i])
        if layouts:
            attrs[_ffi_api.ToAttrKey("input_layouts")] = layouts
    return attrs


def make_relax_conv_bias_pattern(
    op_name: str,
) -> Tuple[relax_pattern.DFPattern, Mapping[str, relax_pattern.DFPattern]]:
    """A simple utility to create patterns for an conv fused with bias.

    Parameters
    ----------
    op_name: str
        The name of a Relax op, such as "relax.nn.conv2d"

    Returns
    -------
    out: tvm.relax.dpl.pattern.DFPattern
        The resulting pattern describing a conv_bias operation.

    annotations: Mapping[str, tvm.relax.dpl.pattern.DFPattern]
        A mapping from name to sub pattern. It can be used to extract
        important expressions from match result, to power the partition
        check function and codegen.
    """

    data = relax_pattern.wildcard()
    weight = relax_pattern.is_const()
    conv = relax_pattern.is_op(op_name)(data, weight)
    bias = relax_pattern.is_const()
    shape = relax_pattern.wildcard()
    reshape = relax_pattern.is_op("relax.reshape")(bias, shape)
    out = relax_pattern.is_op("relax.add")(conv, reshape)
    annotations = {
        "data": data,
        "weight": weight,
        "conv": conv,
        "bias": bias,
        "reshape": reshape,
        "out": out,
    }
    return out, annotations


def _check_relax_conv_bias(context: PatternCheckContext) -> bool:
    """Check if conv_bias fuse pattern is correct.

    Returns
    -------
    pass: bool
        Whether the pattern is correct.
    """

    bias = context.annotated_expr["bias"]
    reshape = context.annotated_expr["reshape"]
    non_one_dims = len([i for i in reshape.struct_info.shape.values if i > 1])
    return non_one_dims <= 1 and bias.struct_info.ndim == 1


def make_relax_linear_pattern() -> (
    Tuple[relax_pattern.DFPattern, Mapping[str, relax_pattern.DFPattern]]
):
    """A simple utility to create patterns for linear.

    Returns
    -------
    out: tvm.relax.dpl.pattern.DFPattern
        The resulting pattern describing a linear operation.

    annotations: Mapping[str, tvm.relax.dpl.pattern.DFPattern]
        A mapping from name to sub pattern. It can be used to extract
        important expressions from match result, to power the partition
        check function and codegen.
    """

    data = relax_pattern.wildcard()
    weight = relax_pattern.is_const()
    permute = relax_pattern.is_op("relax.permute_dims")(weight)
    out = relax_pattern.is_op("relax.matmul")(data, permute)
    annotations = {"data": data, "weight": weight, "permute": permute, "linear": out}
    return out, annotations


def _check_relax_linear(context: PatternCheckContext) -> bool:
    """Check if linear pattern is correct.

    Returns
    -------
    pass: bool
        Whether the pattern is correct.
    """

    weight = context.annotated_expr["weight"]
    permute = context.annotated_expr["permute"]
    return weight.struct_info.ndim == 2 and not permute.attrs["axes"]


def make_relax_linear_bias_pattern() -> (
    Tuple[relax_pattern.DFPattern, Mapping[str, relax_pattern.DFPattern]]
):
    """A simple utility to create patterns for linear with bias.

    Returns
    -------
    out: tvm.relax.dpl.pattern.DFPattern
        The resulting pattern describing a linear_bias operation.

    annotations: Mapping[str, tvm.relax.dpl.pattern.DFPattern]
        A mapping from name to sub pattern. It can be used to extract
        important expressions from match result, to power the partition
        check function and codegen.

    """

    linear, annotations = make_relax_linear_pattern()
    bias = relax_pattern.is_const()
    out = relax_pattern.is_op("relax.add")(linear, bias)
    annotations.update({"bias": bias, "bias_add": out})
    return out, annotations


def _check_relax_linear_bias(context: PatternCheckContext) -> bool:
    """Check if linear_bias pattern is correct.

    Returns
    -------
    pass: bool
        Whether the pattern is correct.
    """

    if not _check_relax_linear(context):
        return False
    bias = context.annotated_expr["bias"]
    return bias.struct_info.ndim == 1


def make_relax_embedding_pattern() -> (
    Tuple[relax_pattern.DFPattern, Mapping[str, relax_pattern.DFPattern]]
):
    """A simple utility to create patterns for embedding.

    Returns
    -------
    out: tvm.relax.dpl.pattern.DFPattern
        The resulting pattern describing a embedding operation.

    annotations: Mapping[str, tvm.relax.dpl.pattern.DFPattern]
        A mapping from name to sub pattern. It can be used to extract
        important expressions from match result, to power the partition
        check function and codegen.
    """

    weight = relax_pattern.is_const()
    data = relax_pattern.wildcard()
    astype = relax_pattern.is_op("relax.astype")(data)
    out = relax_pattern.is_op("relax.take")(weight, astype)
    annotations = {"data": data, "weight": weight, "astype": astype, "take": out}
    return out, annotations


def _check_relax_embedding(context: PatternCheckContext) -> bool:
    """Check if 1d embedding pattern is correct.

    Returns
    -------
    pass: bool
        Whether the pattern is correct.
    """

    weight = context.annotated_expr["weight"]
    astype = context.annotated_expr["astype"]
    return (
        astype.attrs["dtype"] == "int32"
        and weight.struct_info.ndim == 2
        and weight.struct_info.dtype == "float32"
    )


def make_relax_reshape_embedding_pattern() -> (
    Tuple[relax_pattern.DFPattern, Mapping[str, relax_pattern.DFPattern]]
):
    """A simple utility to create patterns for reshaped embedding.

    Returns
    -------
    out: tvm.relax.dpl.pattern.DFPattern
        The resulting pattern describing a reshape_embedding operation.

    annotations: Mapping[str, tvm.relax.dpl.pattern.DFPattern]
        A mapping from name to sub pattern. It can be used to extract
        important expressions from match result, to power the partition
        check function and codegen.
    """

    weight = relax_pattern.is_const()
    data = relax_pattern.wildcard()
    astype = relax_pattern.is_op("relax.astype")(data)
    reduce_shape = relax_pattern.wildcard()
    reduce_in = relax_pattern.is_op("relax.reshape")(astype, reduce_shape)
    take = relax_pattern.is_op("relax.take")(weight, reduce_in)
    expand_shape = relax_pattern.wildcard()
    out = relax_pattern.is_op("relax.reshape")(take, expand_shape)
    annotations = {
        "data": data,
        "weight": weight,
        "astype": astype,
        "reduce_in": reduce_in,
        "take": take,
        "out": out,
    }
    return out, annotations


def _check_relax_reshape_embedding(context: PatternCheckContext) -> bool:
    """Check if reshape embedding pattern is correct.

    Returns
    -------
    pass: bool
        Whether the pattern is correct.
    """

    weight = context.annotated_expr["weight"]
    if weight.struct_info.ndim != 2 or weight.struct_info.dtype != "float32":
        return False
    astype = context.annotated_expr["astype"]
    reduce_in = context.annotated_expr["reduce_in"]
    if astype.attrs["dtype"] != "int32" or reduce_in.struct_info.ndim != 1:
        return False
    return True


def make_relax_attention_pattern() -> (
    Tuple[relax_pattern.DFPattern, Mapping[str, relax_pattern.DFPattern]]
):
    """A simple utility to create patterns for attention.

    Returns
    -------
    out: tvm.relax.dpl.pattern.DFPattern
        The resulting pattern describing a attention operation.

    annotations: Mapping[str, tvm.relax.dpl.pattern.DFPattern]
        A mapping from name to sub pattern. It can be used to extract
        important expressions from match result, to power the partition
        check function and codegen.
    """

    weight_q = relax_pattern.wildcard()
    weight_k = relax_pattern.wildcard()
    weight_v = relax_pattern.wildcard()
    q_trans = relax_pattern.is_op("relax.permute_dims")(weight_q)
    k_trans = relax_pattern.is_op("relax.permute_dims")(weight_k)
    v_trans = relax_pattern.is_op("relax.permute_dims")(weight_v)
    attention = relax_pattern.is_op("relax.nn.attention")(q_trans, k_trans, v_trans)
    out = relax_pattern.is_op("relax.permute_dims")(attention)
    annotations = {
        "weight_q": weight_q,
        "weight_k": weight_k,
        "weight_v": weight_v,
        "q_trans": q_trans,
        "k_trans": k_trans,
        "v_trans": v_trans,
        "attention": attention,
        "out": out,
    }
    return out, annotations


def _check_relax_attention(context: PatternCheckContext) -> bool:
    """Check if attention pattern is correct.

    Returns
    -------
    pass: bool
        Whether the pattern is correct.
    """

    return True


def make_relax_mask_attention_pattern() -> (
    Tuple[relax_pattern.DFPattern, Mapping[str, relax_pattern.DFPattern]]
):
    """A simple utility to create patterns for mask_attention.

    Returns
    -------
    out: tvm.relax.dpl.pattern.DFPattern
        The resulting pattern describing a mask_attention operation.

    annotations: Mapping[str, tvm.relax.dpl.pattern.DFPattern]
        A mapping from name to sub pattern. It can be used to extract
        important expressions from match result, to power the partition
        check function and codegen.
    """

    weight_q = relax_pattern.wildcard()
    weight_k = relax_pattern.wildcard()
    weight_v = relax_pattern.wildcard()
    mask = relax_pattern.wildcard()
    q_trans = relax_pattern.is_op("relax.permute_dims")(weight_q)
    k_trans = relax_pattern.is_op("relax.permute_dims")(weight_k)
    v_trans = relax_pattern.is_op("relax.permute_dims")(weight_v)
    attention = relax_pattern.is_op("relax.nn.attention_bias")(q_trans, k_trans, v_trans, mask)
    out = relax_pattern.is_op("relax.permute_dims")(attention)
    annotations = {
        "weight_q": weight_q,
        "weight_k": weight_k,
        "weight_v": weight_v,
        "mask": mask,
        "q_trans": q_trans,
        "k_trans": k_trans,
        "v_trans": v_trans,
        "attention": attention,
        "out": out,
    }
    return out, annotations


def _check_relax_mask_attention(context: PatternCheckContext) -> bool:
    """Check if mask_attention pattern is correct.

    Returns
    -------
    pass: bool
        Whether the pattern is correct.
    """

    return True


def make_opt_relax_conv_bias_pattern(
    op_name: str,
) -> Tuple[relax_pattern.DFPattern, Mapping[str, relax_pattern.DFPattern]]:
    """Create patterns for an conv2d fused with bias, for mod after optimize.

    Parameters
    ----------
    op_name: str
        The name of a Relax op, such as "relax.nn.conv2d"

    Returns
    -------
    out: tvm.relax.dpl.pattern.DFPattern
        The resulting pattern describing a conv_bias operation.

    annotations: Mapping[str, tvm.relax.dpl.pattern.DFPattern]
        A mapping from name to sub pattern. It can be used to extract
        important expressions from match result, to power the partition
        check function and codegen.
    """

    data = relax_pattern.wildcard()
    weight = relax_pattern.is_const()
    conv = relax_pattern.is_op(op_name)(data, weight)
    bias = relax_pattern.is_const()
    out = relax_pattern.is_op("relax.add")(conv, bias)
    annotations = {"data": data, "weight": weight, "bias": bias, "conv": conv, "out": out}
    return out, annotations


def _check_opt_relax_conv_bias(context: PatternCheckContext) -> bool:
    """Check if conv_bias fuse pattern is correct.

    Returns
    -------
    pass: bool
        Whether the pattern is correct.
    """

    ndim_conv = len(context.annotated_expr["conv"].struct_info.shape.values)
    ndim_bias = len(context.annotated_expr["bias"].struct_info.shape.values)
    ndim_out = len(context.annotated_expr["out"].struct_info.shape.values)
    return ndim_conv == ndim_bias and ndim_bias == ndim_out


def make_opt_relax_linear_pattern() -> (
    Tuple[relax_pattern.DFPattern, Mapping[str, relax_pattern.DFPattern]]
):
    """Create patterns for an linear, for mod after optimize.

    Returns
    -------
    out: tvm.relax.dpl.pattern.DFPattern
        The resulting pattern describing a conv_bias operation.

    annotations: Mapping[str, tvm.relax.dpl.pattern.DFPattern]
        A mapping from name to sub pattern. It can be used to extract
        important expressions from match result, to power the partition
        check function and codegen.
    """

    data = relax_pattern.wildcard()
    weight = relax_pattern.is_const()
    out = relax_pattern.is_op("relax.matmul")(data, weight)
    annotations = {"data": data, "weight": weight, "linear": out}
    return out, annotations


def _check_opt_relax_linear(context: PatternCheckContext) -> bool:
    """Check if linear fuse pattern is correct.

    Returns
    -------
    pass: bool
        Whether the pattern is correct.
    """

    ndim_weight = len(context.annotated_expr["weight"].struct_info.shape.values)
    return ndim_weight == 2


def make_opt_relax_linear_bias_pattern() -> (
    Tuple[relax_pattern.DFPattern, Mapping[str, relax_pattern.DFPattern]]
):
    """Create patterns for an linear_bias, for mod after optimize.

    Returns
    -------
    out: tvm.relax.dpl.pattern.DFPattern
        The resulting pattern describing a conv_bias operation.

    annotations: Mapping[str, tvm.relax.dpl.pattern.DFPattern]
        A mapping from name to sub pattern. It can be used to extract
        important expressions from match result, to power the partition
        check function and codegen.
    """

    data = relax_pattern.wildcard()
    weight = relax_pattern.is_const()
    linear = relax_pattern.is_op("relax.matmul")(data, weight)
    bias = relax_pattern.is_const()
    out = relax_pattern.is_op("relax.add")(linear, bias)
    annotations = {"data": data, "weight": weight, "bias": bias, "linear": linear, "out": out}
    return out, annotations


def _check_opt_relax_linear_bias(context: PatternCheckContext) -> bool:
    """Check if linear fuse pattern is correct.

    Returns
    -------
    pass: bool
        Whether the pattern is correct.
    """

    if not _check_opt_relax_linear(context):
        return False
    ndim_bias = len(context.annotated_expr["bias"].struct_info.shape.values)
    ndim_out = len(context.annotated_expr["out"].struct_info.shape.values)
    return ndim_bias == 1 or ndim_bias == ndim_out


# TODO(tong.meng): support patterns after optimize
register_patterns(
    [
        (
            "msc.conv1d_bias",
            *make_opt_relax_conv_bias_pattern(
                "relax.nn.conv1d",
            ),
            _check_opt_relax_conv_bias,
            partial(msc_attrs_getter, anchor="conv", inputs=["data", "weight", "bias"]),
        ),
        (
            "msc.conv2d_bias",
            *make_opt_relax_conv_bias_pattern(
                "relax.nn.conv2d",
            ),
            _check_opt_relax_conv_bias,
            partial(msc_attrs_getter, anchor="conv", inputs=["data", "weight", "bias"]),
        ),
        (
            "msc.linear",
            *make_opt_relax_linear_pattern(),
            _check_opt_relax_linear,
            partial(msc_attrs_getter, anchor="linear", inputs=["data", "weight"]),
        ),
        (
            "msc.linear_bias",
            *make_opt_relax_linear_bias_pattern(),
            _check_opt_relax_linear_bias,
            partial(msc_attrs_getter, anchor="linear", inputs=["data", "weight", "bias"]),
        ),
        (
            "msc.conv1d_bias",
            *make_relax_conv_bias_pattern(
                "relax.nn.conv1d",
            ),
            _check_relax_conv_bias,
            partial(msc_attrs_getter, anchor="conv", inputs=["data", "weight", "bias"]),
        ),
        (
            "msc.conv2d_bias",
            *make_relax_conv_bias_pattern(
                "relax.nn.conv2d",
            ),
            _check_relax_conv_bias,
            partial(msc_attrs_getter, anchor="conv", inputs=["data", "weight", "bias"]),
        ),
        (
            "msc.linear",
            *make_relax_linear_pattern(),
            _check_relax_linear,
            partial(msc_attrs_getter, anchor="linear", inputs=["data", "weight"]),
        ),
        (
            "msc.linear_bias",
            *make_relax_linear_bias_pattern(),
            _check_relax_linear_bias,
            partial(msc_attrs_getter, anchor="linear", inputs=["data", "weight", "bias"]),
        ),
        (
            "msc.embedding",
            *make_relax_embedding_pattern(),
            _check_relax_embedding,
            partial(msc_attrs_getter, anchor="take", inputs=["data", "weight"]),
        ),
        (
            "msc.embedding",
            *make_relax_reshape_embedding_pattern(),
            _check_relax_reshape_embedding,
            partial(msc_attrs_getter, anchor="take", output="out", inputs=["data", "weight"]),
        ),
        (
            "msc.attention",
            *make_relax_attention_pattern(),
            _check_relax_attention,
            partial(
                msc_attrs_getter, anchor="attention", inputs=["weight_q", "weight_k", "weight_v"]
            ),
        ),
        (
            "msc.attention",
            *make_relax_mask_attention_pattern(),
            _check_relax_mask_attention,
            partial(
                msc_attrs_getter,
                anchor="attention",
                inputs=["weight_q", "weight_k", "weight_v", "mask"],
            ),
        ),
    ]
)
