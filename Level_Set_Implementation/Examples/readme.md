![Logo](/Support_Materials/Assets/Logo_MLJC.png)

<h1 align="center">
  Level Set Implementation - Example
</h1>


This is an example of the level set equation simulation made to illustrate the current capabilities of our model.The fire is set with a circular ignition start. Wind is set constant [-5,-5] in southwest direction. The somewhat triangular shape
expected in this situation is displayed by our model, as shown below.

<p align="center">
  <img src="/Level_Set_Implementation/Examples/fireline.gif">
</p>

Unless otherwise stated, all simulations, referring to the same code for the implementation of the [level set One Fire](/Level_Set_Implementation/Final%20Code/One%20Fire/level_set_OneFire.jl), have this hyper-parameters:

```julia
n = 16
maxIters = 1500
Scale factors = 1
extrapolate = true, timeFactor = 2.5, spatialFactor = 1.25
Lx = 10000   #Domain x width in meters
Ly = 10000   #Domain y width in meters
Lt = 3600    #Total simulation time in seconds

#As usual, the domain is scaled to a cube with corners=10.
#Moreover, the domain is centered at the origin.

xRes = 30    #Resolution of x axis in meters, mesh cell width
yRes = 30    #Resolution of y axis in meters, mesh cell width
tRes = 60    #Time resolution in seconds
```
