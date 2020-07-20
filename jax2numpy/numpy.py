import jax.lax_reference as j_lax_reference
import numpy
import itertools


def neg(object):
    result = -object
    return result


def add(a_object, b_object):
    result = a_object + b_object
    return result


add_any = add


def sub(a_object, b_object):
    result = a_object - b_object
    return result


def mul(a_object, b_object):
    result = a_object * b_object
    return result


def div(a_object, b_object):
    result = a_object / b_object
    return result


def pow(object, y=None):
    result = object ** y
    return result


integer_pow = pow


def log(object):
    result = numpy.log(object)
    return result


def eq(a_object, b_object):
    result = a_object == b_object
    return result


def tie_in(a_object, b_object):
    return b_object


def convert_element_type(object, new_dtype=None, old_dtype=None):
    result = numpy.asarray(object, dtype=new_dtype)
    return result


broadcast_in_dim = j_lax_reference.broadcast_in_dim


def select(condition, x, y):
    result = numpy.where(condition, x, y)
    return result


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


def _gather(
    operand,
    start_indices,
    index_vector_dim,
    offset_dims,
    slice_sizes,
    collapsed_slice_dims,
    start_index_map,
):
    result_shape = []
    batch_dims = []

    adjusted_slice_sizes = [
        slice_size
        for index, slice_size in enumerate(slice_sizes)
        if index not in collapsed_slice_dims
    ]

    for index in itertools.count():
        try:
            k = offset_dims.index(index)

        except ValueError:
            pass

        else:
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
        if index not in collapsed_slice_dims
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
        Out = numpy.meshgrid(
            *[list(range(length)) for length in result_shape], indexing="ij"
        )  # create index arrays
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


def scatter_add(
    operand,
    scatter_indices,
    updates,
    update_jaxpr=None,
    update_consts=None,
    dimension_numbers=None,
):
    (
        update_window_dims,
        inserted_window_dims,
        scatter_dims_to_operand_dims,
    ) = dimension_numbers
    result = _scatter_add(
        operand,
        scatter_indices,
        updates,
        add,  # update_jaxpr,
        -1,
        update_window_dims,
        inserted_window_dims,
        scatter_dims_to_operand_dims,
    )
    return result


def _scatter_add(
    operand,
    scatter_indices,
    updates,
    update_computation,
    index_vector_dim,
    update_window_dims,
    inserted_window_dims,
    scatter_dims_to_operand_dims,
):
    update_scatter_dims = sorted(
        set(range(len(updates.shape))) - set(update_window_dims)
    )

    window_dims_to_operand_dims = [
        index
        for index in range(len(operand.shape))
        if index not in inserted_window_dims
    ]

    result = numpy.copy(operand)

    for U in itertools.product(*[range(length) for length in updates.shape]):
        G = [U[k] for k in update_scatter_dims]
        G.insert(index_vector_dim, slice(None))
        S = scatter_indices[tuple(G)]
        S_in = numpy.zeros((len(operand.shape),), dtype=int)
        S_in[scatter_dims_to_operand_dims] = S
        W_in = numpy.zeros((len(operand.shape),), dtype=int)
        W_in[[window_dims_to_operand_dims[index] for index in update_window_dims]] = [
            U[index] for index in update_window_dims
        ]
        I = W_in + S_in

        result[tuple(I)] = update_computation(result[tuple(I)], updates[U])

    return result
