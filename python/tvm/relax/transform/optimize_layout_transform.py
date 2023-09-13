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
# pylint: disable=invalid-name, unused-argument, redefined-argument-from-local
"""Relax Optimize Layout Transform pass."""
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relax import Expr, Function
from tvm.relax.dpl import is_op, rewrite_call, wildcard
from . import function_pass


@function_pass(opt_level=0)
class OptimizeLayoutTransform:
    """
    Pass to remove redundant transform layout operators
    introduced by AlterOpImpl pass.
    """

    def __init__(self):
        self.input = wildcard()
        pattern_transform_layout = is_op("relax.layout_transform")(self.input)
        pattern_1 = is_op("relax.layout_transform")(pattern_transform_layout)

        self.pattern = pattern_1

    def transform_function(self, func: Expr, mod: IRModule, ctx: PassContext) -> IRModule:
        """
        Tranformation function to pattern match layout_transform -> layout_transform
        pattern

        Parameters
        ----------
        func: Expr
            The relax function to be optimized

        mod: IRModule
            The ir module

        ctx: PassContext
            Relax pass context
        """

        updated_func = func
        for _, func in mod.functions.items():
            # Skip non-relax functions
            if not isinstance(func, Function):
                continue
            # Skip primitive functions
            if "Primitive" in func.attrs.keys() and func.attrs["Primitive"] != 0:
                continue

            def rewriter(expr, matches):
                arg1 = matches[self.pattern]
                arg2 = matches[self.input]
                if list(arg1.struct_info.shape) == list(arg2.struct_info.shape):
                    return arg2
                return expr

            updated_func = rewrite_call(self.pattern, rewriter, func)

        return updated_func
