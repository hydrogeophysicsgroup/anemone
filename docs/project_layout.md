# The structure of anemone

Anemone consists of four modules.

1. **forwards** - where everything comes together.
2. **experiment** - objects representing experiment configuration parameters like transmit loop geometry, transmitter waveform, receiver filters, etc.
3. **transforms** - operators for the numerical computation of Hankel transforms, Fourier transforms and Inverse Laplace transforms.
4. **physics** - base routines like reflection coefficients, wave numbers, etc.


We will describe each of these modules here.



## Forwards


## Experiment

A module containing various python objects representing experimental setup parameters.

The `Loop` object contains the setup of the transmitter loop.
For example, a 3m x 3m transmit loop can be instantiated with the call:
```python
tx = Loop.square(3)
```

## Transforms

The Hankel transform.


## Physics

This is pretty deep in the weeds, for your information.
The physics itself make reference to Ward & Hohmann 1988 including the equation and page numbers.

An important part of the physic module design is that functions are agnostic to the input array dimensionality. In this manner, functions closely resemble their mathematical formation. However, this means the dimensionality of all the input arrays must be compatible to begin with.
Throughout the package, we use pytorch's (orignally numpy's ?) array broadcasting to accomplish this.

The peak of the physics module is a the functions the compute the reflection coefficients for transverse electric and transverse magnetic waves. These have the function signature

```python
rte = refl_transverse_electric(sigma, thickness, omega, wave_length)
```
The order of the arguments, matches the necessary order of the dimensions. so for a 10 layer sigma, 9 thicknesses, 101 frequencies, and 301 wave lengths, the dimensions will need to be

| tensor | dimensions |
|:------:|:----------:|
| `sigma`       | (10, 1, 1)  |
| `thick`       | (9, 1, 1)   |
| `omega`       | (1, 101, 1) |
| `wave_length` | (1, 1, 301) |
| return value: `rte`         | (1, 101, 301) |

This is necessary in order for the tensor to multiply and add with one another.
Pytorch has a very nice notation to accomplish this. If all the tensors are 1 dimensional a function call can look like this:

```python
rte = refl_transverse_electric(sigma[:,None,None], 
                               thick[:,None,None], 
                               omega[None, :,None], 
                               wave_length[None, None, :])
```
This will return the reflection coefficients at the surface with a dimensionality of (1, 101, 301).
In this manner, all the arrays are broadcast to the necessary dimensions. If any of the arrays are floats, no broadcasting is needed.