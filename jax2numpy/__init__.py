import jax2numpy.ast as j_ast
import jax2numpy.numba as j_numba
import jax2numpy.numpy as j_numpy
import astor
import black
import jax.abstract_arrays as j_abstract_arrays
import jax.api_util as j_api_util
import jax.core as j_core
import jax.interpreters.partial_eval as ji_partial_eval
import jax.interpreters.xla as ji_xla
import jax.lax.lax as jl_lax
import jax.linear_util as j_linear_util
import jax.tree_util as j_tree_util
import jax.util as j_util
import numba
import numpy
import traceback

import jax.config as j_config

j_config.update("jax_enable_x64", True)  # necessary for asserts


def get_function(function, args=None, kwargs=None, catch_numba=True):
    jaxpr, constants = _get_jax_objects(
        function, tuple() if args is None else args, {} if kwargs is None else kwargs
    )

    print()
    print("jaxpr")
    print(repr(jaxpr))
    print("constants")
    print(repr(constants))

    ast_builder = AstBuilder()

    name_node = ast_builder.get_ast_node(jaxpr, constants)
    module_node = ast_builder.get_module()

    print("module")
    print(_get_code_string(module_node))

    environment = vars(j_numpy).copy()
    exec(compile(module_node, "<string>", mode="exec"), environment)
    numpy_function = environment[name_node.id]

    environment = vars(j_numba).copy()
    exec(compile(module_node, "<string>", mode="exec"), environment)
    numba_function = numba.jit(environment[name_node.id])

    def result(*args, **kwargs):
        expected = function(*args, **kwargs)

        arguments, _ = j_tree_util.tree_flatten((args, kwargs))

        (result,) = j_core.eval_jaxpr(jaxpr, constants, *arguments)
        assert _are_equal(result, expected)

        (result,) = numpy_function(*args, **kwargs)
        assert _are_equal(result, expected)

        try:
            (result,) = numba_function(*args, **kwargs)

        except:
            if not catch_numba:
                raise

            traceback.print_exc()

        else:
            assert _are_equal(result, expected)

        return expected

    return result


def _get_jax_objects(function, args, kwargs):
    # Set up function for transformation
    wrapped_function = j_linear_util.wrap_init(function)
    # Flatten input arguments
    jax_arguments, in_tree = j_tree_util.tree_flatten((args, kwargs))
    # Transform function to accept flat arguments
    # and return a flat list result
    jaxtree_function, _ = j_api_util.flatten_fun(wrapped_function, in_tree)
    # Abstract and partial-value's flat arguments
    partial_values = j_util.safe_map(_get_partial_value, jax_arguments)
    # Trace function into Jaxpr
    jaxpr, _, constants = ji_partial_eval.trace_to_jaxpr(
        jaxtree_function, partial_values
    )

    result = (jaxpr, constants)
    return result


def _get_partial_value(object):
    # ShapedArrays are abstract values that carry around
    # shape and dtype information
    aval = j_abstract_arrays.ShapedArray(numpy.shape(object), numpy.result_type(object))
    result = ji_partial_eval.PartialVal((aval, j_core.unit))
    return result


def _get_code_string(node):
    result = black.format_str(astor.to_source(node), mode=black.FileMode())
    return result


def _are_equal(a_object, b_object):
    if isinstance(a_object, (numpy.ndarray, ji_xla.DeviceArray)) and isinstance(
        b_object, (numpy.ndarray, ji_xla.DeviceArray)
    ):
        result = numpy.array_equal(a_object, b_object)
        return result

    result = a_object == b_object
    return result


_get_node_function_by_primitive_name = {}


class AstBuilder:
    def __init__(self):
        self._function_name_node_generator = _FunctionNameNodeGenerator()
        self._name_node_generator = _NameNodeGenerator()
        self._function_definition_nodes = {}
        self._constant_cache = {}

    def get_ast_node(self, jaxpr, constants):
        function_name_node = self._function_name_node_generator.get_name_node(jaxpr)

        if id(jaxpr) in self._function_definition_nodes.keys():
            return function_name_node

        body_nodes = []

        for equation in jaxpr.eqns:
            get_ast = _get_node_function_by_primitive_name.get(
                equation.primitive.name, None
            )
            if get_ast is not None:
                raise Exception((equation.invars, equation.params))
                sub_result = get_ast()

            else:
                arg_nodes = [
                    self._get_variable_node(in_variable, jaxpr.constvars, constants)
                    for in_variable in equation.invars
                ]
                kwarg_nodes = {
                    name: _get_literal_node(object, self)
                    for name, object in equation.params.items()
                }
                sub_result = j_ast.get_call(
                    j_ast.parse(self._get_variable_name(equation.primitive.name)),
                    args=arg_nodes,
                    kwargs=kwarg_nodes,
                )

            if equation.primitive.multiple_results:
                assignment_node = j_ast.parse(
                    "{target} = {result}",
                    substitutes=dict(
                        target=j_ast.get_tuple(
                            [
                                self._name_node_generator.get_name_node(
                                    repr(out_variable), context=j_ast.STORE_CONTEXT
                                )
                                for out_variable in equation.outvars
                            ],
                            j_ast.STORE_CONTEXT,
                        ),
                        result=sub_result,
                    ),
                )

            else:
                assignment_node = j_ast.parse(
                    "{target} = {result}",
                    substitutes=dict(
                        target=self._name_node_generator.get_name_node(
                            repr(equation.outvars[0]), context=j_ast.STORE_CONTEXT
                        ),
                        result=sub_result,
                    ),
                )

            body_nodes.append(assignment_node)

        return_node = j_ast.parse(
            "return {result}",
            substitutes=dict(
                result=j_ast.get_tuple(
                    [
                        self._get_variable_node(
                            out_variable, jaxpr.constvars, constants
                        )
                        for out_variable in jaxpr.outvars
                    ],
                    j_ast.LOAD_CONTEXT,
                )
            ),
        )
        body_nodes.append(return_node)

        function_definition_node = j_ast.get_function_definition(
            function_name_node.id,
            body_nodes,
            argument_names=[repr(in_variable) for in_variable in jaxpr.invars],
        )
        self._function_definition_nodes[id(jaxpr)] = function_definition_node

        return function_name_node

    def _get_variable_name(self, name):
        result = name.replace("-", "_")
        return result

    def _get_variable_node(self, in_variable, constant_variables, constant_values):
        result = None

        if isinstance(in_variable, j_core.Literal):
            constant_value = in_variable.val

        else:
            try:
                index = constant_variables.index(in_variable)

            except ValueError:
                if repr(in_variable) == "*":
                    result = self._get_constant(None)

                else:
                    result = self._name_node_generator.get_name_node(repr(in_variable))

            else:
                constant_value = constant_values[index]

        if result is None:
            result = self._get_constant(constant_value)

        return result

    def _get_constant(self, value):
        key = id(value)

        result = self._constant_cache.get(key, None)
        if result is not None:
            return result

        result = _get_literal_node(value, self)
        self._constant_cache[key] = result
        return result

    def get_module(self):
        result = j_ast.get_module(self._function_definition_nodes.values())
        return result


def _get_literal_node(object, ast_builder):
    if isinstance(object, numpy.ndarray):
        if len(object.shape) == 0:
            result = _get_literal_node(
                object.tolist(), ast_builder
            )  # WARNING could be an issue
            return result

        result = j_ast.parse(
            "numpy.array({data}, dtype={dtype})",
            substitutes=dict(
                data=j_ast.parse(repr(object.tolist())),
                dtype=_get_literal_node(object.dtype, ast_builder),
            ),
        )
        return result

    if isinstance(object, numpy.dtype):
        result = j_ast.parse(
            "numpy.dtype({name})", substitutes=dict(name=j_ast.parse(repr(object.name)))
        )
        return result

    if isinstance(
        object, (int, float, numpy.int32, numpy.int64, numpy.float32)
    ):  # WARNING numpy.int32, numpy.float32 could be an issue
        result = j_ast.parse(repr(object))
        return result

    if isinstance(object, ji_xla.DeviceArray):
        result = _get_literal_node(numpy.asarray(object), ast_builder)
        return result

    if object is None:
        result = j_ast.parse("None")
        return result

    if isinstance(object, jl_lax.GatherDimensionNumbers):
        result = _get_literal_node(
            (object.offset_dims, object.collapsed_slice_dims, object.start_index_map),
            ast_builder,
        )
        return result

    if isinstance(object, jl_lax.ScatterDimensionNumbers):
        result = _get_literal_node(
            (
                object.update_window_dims,
                object.inserted_window_dims,
                object.scatter_dims_to_operand_dims,
            ),
            ast_builder,
        )
        return result

    if isinstance(object, tuple):
        result = j_ast.get_tuple(
            [_get_literal_node(element, ast_builder) for element in object],
            j_ast.LOAD_CONTEXT,
        )
        return result

    if isinstance(object, j_core.Jaxpr):
        result = ast_builder.get_ast_node(object, {})
        return result

    raise NotImplementedError(type(object))


class _FunctionNameNodeGenerator:
    def __init__(self):
        self._index = 0
        self._cache = {}

    def get_name_node(self, jaxpr):
        key = id(jaxpr)

        result = self._cache.get(key, None)
        if result is not None:
            return result

        self._index += 1
        result = j_ast.parse("_f{}".format(self._index))
        self._cache[key] = result
        return result


class _NameNodeGenerator:
    KEYWORD_STRINGS = ["if", "in", "is", "or", "def", "del", "for", "jax"]

    def __init__(self):
        self._cache = {}

    def get_name_node(self, name, context=j_ast.LOAD_CONTEXT):
        key = (name, context)

        result = self._cache.get(key, None)
        if result is not None:
            return result

        if name in _NameNodeGenerator.KEYWORD_STRINGS:
            name = "{}_".format(name)

        result = j_ast.parse(name)
        self._cache[key] = result
        return result
