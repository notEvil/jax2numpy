import ast
import itertools


LOAD_CONTEXT = ast.Load()
STORE_CONTEXT = ast.Store()


def parse(string, substitutes=None):
    if substitutes is None:
        substitutes = {}

    variable_names = iter("_a{}".format(_) for _ in itertools.count(1))
    placeholder = {}

    for name in substitutes.keys():
        while True:
            variable_name = next(variable_names)

            if variable_name not in string:
                break

        placeholder[name] = variable_name

    node = ast.parse(string.format(**placeholder), mode="exec")
    if len(node.body) == 1:
        (node,) = node.body
        if isinstance(node, ast.Expr):
            node = node.value

    substitutes = {
        variable_name: substitutes[name] for name, variable_name in placeholder.items()
    }

    replacer = _PlaceholderReplacer(substitutes)
    result = replacer.visit(node)

    return result


class _PlaceholderReplacer(ast.NodeTransformer):
    def __init__(self, substitutes):
        super().__init__()

        self.substitutes = substitutes

    def visit_Name(self, node):
        substitute = self.substitutes.get(node.id, None)

        if substitute is None:
            return node

        result = _replace_context(substitute, node.ctx)
        return result


class _ContextReplacer(ast.NodeTransformer):
    def __init__(self, context):
        super().__init__()

        self.context = context

    def visit_Name(self, node):
        node.ctx = self.context
        return node

    def visit_Subscript(self, node):
        node.ctx = self.context
        return node


_load_context_replacer = _ContextReplacer(LOAD_CONTEXT)
_store_context_replacer = _ContextReplacer(STORE_CONTEXT)


def _replace_context(node, context):
    context_replacer = (
        _load_context_replacer
        if isinstance(context, ast.Load)
        else _store_context_replacer
    )
    result = context_replacer.visit(node)
    return result


def get_call(function_node, args=None, kwargs=None, line_number=0, column_offset=0):
    result = ast.Call(
        func=function_node,
        args=[] if args is None else args,
        keywords=[]
        if kwargs is None
        else [ast.keyword(arg=name, value=value) for name, value in kwargs.items()],
        lineno=line_number,
        col_offset=column_offset,
    )
    return result


def get_function_definition(
    name,
    body_nodes,
    argument_names=None,
    args_name=None,
    line_number=0,
    column_offset=0,
):
    if argument_names is None:
        argument_names = []

    args_2 = ast.arguments(
        args=[
            ast.arg(
                arg=argument_name,
                lineno=line_number,
                col_offset=column_offset,
                annotation=None,
            )
            for argument_name in argument_names
        ],
        vararg=None
        if args_name is None
        else ast.arg(arg=args_name, lineno=line_number, col_offset=column_offset),
        posonlyargs=[],
        defaults=[],
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
    )

    result = ast.FunctionDef(
        name=name,
        args=args_2,
        body=body_nodes,
        decorator_list=[],
        lineno=line_number,
        col_offset=column_offset,
    )
    return result


def get_module(body_nodes, line_number=0, column_offset=0):
    result = ast.Module(
        body=list(body_nodes),
        type_ignores=[],
        lineno=line_number,
        col_offset=column_offset,
    )
    return result


def get_tuple(items, context, line_number=0, column_offset=0):
    result = ast.Tuple(
        elts=list(items), ctx=context, lineno=line_number, col_offset=column_offset
    )
    return result
