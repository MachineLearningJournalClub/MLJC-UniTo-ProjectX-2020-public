![Logo](/Support_Materials/Assets/Logo_MLJC.png)

<h1 align="center">
  Interpolation
</h1>

The current implementation of the PINNs architecture requires the vector / scalar fields of the input parameters (for example wind, altimetric profile, fuel etc.) to be loaded in the form of differentiable functions. This is due to some limitations of the library NeuralPDE.jl (that currently is still under development).
In the practial applications, these map are in the form of matrices of data, for this reason to carry out the simulations presented in this proposal we created some scripts that perform curve fitting of the discretized data using polynomial and gaussians.

A first [interpolation example](/BC_Interpolation/Interpolation_Example/Surface_fit.cpp) is shown below. 
<p align="center">
  <img src="/BC_Interpolation/Interpolation_Example/ExampleScalar2D.png">
</p>

After this example, we built the interpolation of the altitude profile's gradient of Isom Creek fire through following stages:

* [Extrapolation of altitude profile's gradient](/BC_Interpolation/IsomCreek_Alt/IsomCreek_altimetric_preprocessing.ipynb) from netCDF files([geo_em.d01](/BC_Interpolation/IsomCreek_Alt/geo_em.d01.nc), [geo_em.d02](/BC_Interpolation/IsomCreek_Alt/geo_em.d02.nc));
* [Altitude fit](/BC_Interpolation/IsomCreek_Alt/alt_fit.cpp) in CERN root. We performed the fit using non linear minimizers in order to find the parameters of a 2D polynomial functional. Then the gradient is computed symbolically. 
