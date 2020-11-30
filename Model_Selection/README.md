![Logo](/Support_Materials/Assets/Logo_MLJC.png)
<h1 align="center">
  Model Selection
</h1>


### [Lotka-Volterra](/Model_Selection/DiffEqFlux/ODE%20Lotka-Volterra%20DiffEqFlux.jl)

Following Chris Rackauckas' approach [[1]](#1), we reimplemented the following Lotka-Volterra equations, embedding the ODE into a neural network using Julia's library *DiffEqFlux*.

 ### [Neural ODE 1D](/Model_Selection/DiffEqFlux/NeuralODE1D%20DiffEqFlux.jl)

 ### Diffusion Equation:defining a 2-D reaction-diffusion equation. It describes combustion dynamics, similar to WRF-Fire equations.

 + #### [Diffusion Equation with a NN](/Model_Selection/DiffEqFlux/DiffusionEquation_NN.jl)
 + #### [Diffusion Equation with a CNN](/Model_Selection/DiffEqFlux/DiffusionEquation_CNN.jl)
 + #### [Numerical Diffusion Equation](/Model_Selection/DiffEqFlux/DiffusionEquationNumericalBenchmark.jl)
 + #### [Diffusion Equation NeuralBenchmark](/Model_Selection/DiffEqFlux/DiffusionEquationNeuralBenchmark.jl). We employ an UPDE solved based on a NN combined with a CNN.

 ### Fisher KPP
 + #### [Fisher KPP example with CNN resolution](/Model_Selection/DiffEqFlux/FisherKPPexample1D.jl)
 + #### [Fisher KPP example with RNN resolution](/Model_Selection/DiffEqFlux/FisherKPPexample1D_pipeline(NN,CNN,RNN).jl)
 
 ### [Heat equation](/Model_Selection/NeuralPDE/heat_equation.jl)
 
 ### [Poisson's Equation](/Model_Selection/NeuralPDE/poisson2D.jl)
 
 ### Burger's equation
 + #### [Burger's equation](/Model_Selection/DiffEqFlux/Burger.jl)
 + #### [Burger's equation: comparison between neural and numerical approach](/Model_Selection/NeuralPDE/burger_neural_vs_num.jl)
 + #### [Non Linear Burger's equation](/Model_Selection/NeuralPDE/BurgerNonLinear.jl)
 



###  References
<a id="1">[1]</a>
Chris Rackauckas et al (2019).
[_DiffEqFlux.jl - A Julia Library for Neural Differential Equations_](https://arxiv.org/abs/1902.02376)
