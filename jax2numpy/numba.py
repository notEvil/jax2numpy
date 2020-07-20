import jax2numpy.numpy as j_numpy
import numba
import numpy
import typing


_jit = numba.jit(nopython=True)


for name in ["add", "add_any", "mul", "div", "pow", "integer_pow"]:
    locals()[name] = _jit(getattr(j_numpy, name))


@_jit
def gather(operand, start_indices, dimension_numbers, slice_sizes):
    offset_dims, collapsed_slice_dims, start_index_map = dimension_numbers
    result = _gather(
        operand,
        start_indices,
        -1,
        offset_dims,
        slice_sizes,
        collapsed_slice_dims,
        start_index_map,
    )
    return result


@_jit
def _gather(
    operand,
    start_indices,
    index_vector_dim,
    offset_dims: typing.Tuple[int, ...],
    slice_sizes,
    collapsed_slice_dims: typing.Tuple[int, ...],
    start_index_map,
):
    offset_dims = numpy.array(offset_dims)
    collapsed_slice_dims = numpy.array(collapsed_slice_dims)

    result_shape = []
    batch_dims = []

    adjusted_slice_sizes = [
        slice_size
        for index, slice_size in enumerate(slice_sizes)
        # if index not in collapsed_slice_dims
        if not _in(index, collapsed_slice_dims)
    ]

    # for index in itertools.count():
    index = -1
    while True:
        index += 1

        # try:
        #     k = offset_dims.index(index)
        #
        # except ValueError:
        #     pass
        #
        # else:
        k = _index(index, offset_dims)
        if k != -1:
            _ = adjusted_slice_sizes[k]
            result_shape.append(_)
            continue

        k = len(batch_dims)  # WARNING assumed
        batch_dims.append(index)

        _ = k if k < index_vector_dim else (k + 1)
        if len(start_indices.shape) <= _:  # WARNING assumed
            batch_dims.pop()
            break
        _ = start_indices.shape[_]
        result_shape.append(_)

    remapped_offset_dims = [
        index
        for index in range(len(operand.shape))
        # if index not in collapsed_slice_dims
        if not _in(index, collapsed_slice_dims)
    ]

    if False:
        result = numpy.ndarray(result_shape)

        for Out in itertools.product(*[range(length) for length in result_shape]):
            G = [Out[k] for k in batch_dims]
            G.insert(index_vector_dim, slice(None))
            S = start_indices[tuple(G)]
            S_in = numpy.zeros((len(operand.shape),), dtype=int)
            S_in[start_index_map] = S
            O_in = numpy.zeros((len(operand.shape),), dtype=int)
            O_in[remapped_offset_dims] = [Out[index] for index in offset_dims]
            In = O_in + S_in
            result[Out] = operand[tuple(In)]

    if True:
        # Out = numpy.meshgrid(
        #     *[list(range(length)) for length in result_shape], indexing="ij"
        # )  # create index arrays
        Out = _meshgrid(result_shape)
        return 1  # TODO remove
        Out = [numpy.ravel(_) for _ in Out]  # flatten index arrays
        G = [Out[k] for k in batch_dims]
        G.insert(index_vector_dim, slice(None))
        S = start_indices[tuple(G)]  # get start indices
        if index_vector_dim != 0:  # fix shape
            S = numpy.transpose(S)
        In = [0] * len(operand.shape)

        for _, s in zip(start_index_map, S):
            In[_] = s
        for index, remapped_index in zip(offset_dims, remapped_offset_dims):
            In[remapped_index] = In[remapped_index] + Out[index]
        result = numpy.reshape(operand[tuple(In)], result_shape)

    return result


@_jit
def _in(object, array):
    result = (array == object).any()
    return result


@_jit
def _index(object, array):
    for index, _ in enumerate(array):
        if _ == object:
            return index

    return -1


@_jit
def _meshgrid(lengths):
    shape = tuple(lengths[::-1])  # can't create tuples in Numba :(
    length = numpy.prod(numpy.array(shape))
    a = 1
    for length in lengths:
        _ = numpy.repeat(numpy.repeat(numpy.arange(length), a), length // a)
        _ = numpy.reshape(_, shape)
        a *= length

    return 1  # TODO remove
