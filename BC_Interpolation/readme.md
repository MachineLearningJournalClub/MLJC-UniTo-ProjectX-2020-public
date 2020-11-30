![Logo](/Support_Materials/Assets/Logo_MLJC.png)

<h1 align="center">
  Interpolation
</h1>

We built an interpolation model of discrete-valued fields using functional forms. Our aim is to train neural networks that behave like continuos functions.

In particular, we built the interpolation of the altitude profile's gradient of Isom Creek fire through following stages:

* [Extrapolation of altitude profile's gradient]() from netCDF files([geo_em.d01](), [geo_em.d02]());
* [Altitude fit]() in CERN root. We performed the fit using non linear minimizers in order to find the parameters of a 2D polynomial functional. Then the gradient is computed symbolically. 
