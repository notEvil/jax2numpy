# jax2numpy

This is work in progress with currently no ambition to be usable!

## What is it

My attempt to combine the transformations of Jax with the compiler of Numba. See https://github.com/google/jax/issues/2126 . It transforms a Jaxpr into an ast node.

## Issues

- Numba doesn't support dynamic tuple creation
-- but requires tuples for many Numpy functions
- Numba doesn't support any of Numpys broadcasting functions
