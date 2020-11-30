# Work in progress...

 <img src="https://www.provincia.vicenza.it/immagini/work_in_porgress_.jpg/image" alt="Work in progress" width="200" height="200"> 
 


![Logo](/Support_Materials/Assets/Logo_MLJC.png)

<h1 align="center">
  Physics-Informed Machine Learning Simulator for Wildfire Propagation
</h1>

The aim of this work is to evaluate the feasibility of re-implementing some key parts of the widely used Weather Research and Forecasting WRF-SFIRE simulator by replacing its core differential equations numerical solvers with state-of-the-art physics-informed machine learning techniques to solve ODEs and PDEs, in order to transform it into a real-time simulator for wildfire spread prediction. Our ML approach is based on Physics Informed Neural Networks implemented in the [NeuralPDE.jl](https://github.com/SciML/NeuralPDE.jl) package, which turns an integration problem into a minimization one.  

### [Model Selection](/Model_Selection/README.md)
A collection of our attempts to find a proper model which suites all of our needs. We have ex-plored different architectures within the field of Scientific Ma-chine Learning.  We started our investigations using theDiffE-qFlux.jllibrary, which defines and solves neural ordinarydifferential equations (i.e. ODEs where a neural network defines its derivative function)

### [Interpolation](/Interpolation/readme.md)
Some utility notebooks needed to implement key informations (terrain slope, wind field) in our model. It also provides better perfomances in terms of speed and computational load.

### [WRF]()
The results obtained by our simulation performed with the Weather Research Forecast system. Firstly we have done a profiling with the perf tool, in order to mesure the overhead of WRF's subroutines. Later we run several simulations of fire and atmospherical events. We then kept the result for the Isom Creek and OneFire cases.

### [Level Set Implementation](/Level_Set_Implementation/readme.md)
The level-set is the mathematical core for calculating the spread of the fire.  The minimization of the loss func-tions is the process that actually solves the PDE and constitutesthe  main  load  for  the  CPU.  It  can  be  easily  accelerated  usingGPUs. The  model  was  implemented  using  the  low-level  interface of   the NeuralPDE.jl library which contains the necessarymethods for the generation of the training datasets and of theloss functions starting from the explicit form of the equations and the boundary conditions.

### [Euler System Implementation]()

### [Julia Environments](/Julia_Environments/readme.md)






