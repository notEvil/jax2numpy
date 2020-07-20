import jax2numpy
import jax
import numpy


settings = []


def function(argument):
    result = 3 * argument ** 2 + 2 * argument + 1
    return result


settings.append((function, (1.2,), None))
settings.append((jax.grad(function), (1.2,), None))
settings.append((jax.vmap(function), (numpy.array([1.2]),), None))
settings.append((jax.vmap(jax.grad(function)), (numpy.array([1.2]),), None))


def function(argument):
    result = (argument[0] / argument[1]) ** (argument[2])
    return result


settings.append((function, (numpy.array([1.2, 2.3, 3.4]),), None))
settings.append((jax.grad(function), (numpy.array([1.2, 2.3, 3.4]),), None))
settings.append((jax.vmap(function), (numpy.array([[1.2, 2.3, 3.4]]),), None))
settings.append((jax.vmap(jax.grad(function)), (numpy.array([[1.2, 2.3, 3.4]]),), None))


for function, args, kwargs in settings:
    if kwargs is None:
        kwargs = {}

    numpy_function = jax2numpy.get_function(
        function, args=args, kwargs=kwargs, catch_numba=True
    )
    print(numpy_function(*args, **kwargs))
