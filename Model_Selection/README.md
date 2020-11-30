![Logo](./Visualization_Model selection/Logo MLJC.png)
<h1 align="center">
  Model Selection
</h1>


### [Lotka-Volterra](./ODE Lotka-Volterra DiffEqFlux.jl)

Following Chris Rackauckas' approach [[1]](#1), we reimplemented the following Lotka-Volterra equations, embedding the ODE into a neural network using Julia's library *DiffEqFlux*.

The Lotka-Volterra equations, also called predator-prey equations, are a system of first-order nonlinear differential equations, describing the interaction between two species, usually one seen as a predator, the other as a prey.  

<img src="https://latex.codecogs.com/svg.latex?\begin{align*}&space;&&space;\frac{\mathrm{d}&space;x&space;}{\mathrm{d}&space;t}&space;=&space;\alpha&space;x&space;-&space;\beta&space;x&space;y,&space;\\&space;&&space;\frac{\mathrm{d}&space;y&space;}{\mathrm{d}&space;t}&space;=&space;\delta&space;xy&space;-&space;\beta&space;x&space;y.&space;\end{align*}" title="\begin{align*} & \frac{\mathrm{d} x }{\mathrm{d} t} = \alpha x - \beta x y, \\ & \frac{\mathrm{d} y }{\mathrm{d} t} = \delta xy - \beta x y. \end{align*}" />

&nbsp;

We first solved the equations numerically, using Julia's library *DifferentialEquations* and we plotted the solution, shown below.

! [Numerical solution Lotka-Volterra](./Visualization_Model selection/Numerical solution Lotka-Volterra.png)

Then we used the previous numerical solution to generate data (plotted above) in order to train through *Flux* library the neural network.

 ! [Data_points Lotka-Volterra](./Visualization_Model selection/Data_points Lotka-Volterra.png)

 ### [Neural ODE 1D](NeuralODE1D DiffEqFlux)

 ### [Diffusion Equation]() Defining a 2-D reaction-diffusion equation. It describes combustion dynamics, similar to WRF-Fire equations.

 + #### [Diffusion Equation with a NN]()
 + #### [Diffusion Equation with a CNN]()
 + #### [Numerical Diffusion Equation]()
 + #### [Diffusion Equation NeuralBenchmark]() We employ an UPDE solved based on a NN combined with a CNN.

 ### [Fisher KPP]()
 + #### [Fisher KPP example]()
 + #### [Fisher KPP example with RNN resolution]()



###  References
<a id="1">[1]</a>
Chris Rackauckas et al (2019).
[DiffEqFlux.jl - A Julia Library for Neural Differential Equations](https://arxiv.org/abs/1902.02376)
