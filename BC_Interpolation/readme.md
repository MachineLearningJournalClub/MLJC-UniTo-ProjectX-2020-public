![Logo](/Support_Materials/Assets/Logo_MLJC.png)

<h1 align="center">
  Interpolation
</h1>

We built an interpolation model of discrete-valued fields using functional forms. Our aim is to train neural networks that behave like continuos functions.

A first [interpolation example](/BC_Interpolation/Interpolation_Example/Surface_fit.cpp) is shown below. 
<p align="center">
  <img src="/BC_Interpolation/Interpolation_Example/ExampleScalar2D.png">
</p>

After this example, we built the interpolation of the altitude profile's gradient of Isom Creek fire through following stages:

* [Extrapolation of altitude profile's gradient](/BC_Interpolation/IsomCreek_Alt/IsomCreek_altimetric_preprocessing.ipynb) from netCDF files([geo_em.d01](/BC_Interpolation/IsomCreek_Alt/geo_em.d01.nc), [geo_em.d02](/BC_Interpolation/IsomCreek_Alt/geo_em.d02.nc));
* [Altitude fit](/BC_Interpolation/IsomCreek_Alt/alt_fit.cpp) in CERN root. We performed the fit using non linear minimizers in order to find the parameters of a 2D polynomial functional. Then the gradient is computed symbolically. 
