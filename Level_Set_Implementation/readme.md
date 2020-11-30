<h1 align="center">
  Level Set Implementations
</h1>

The level-set is the mathematical core for calculating the spread of the fire. The minimization of the loss functions is the process that actually solves the PDE and constitutesthe main load for the CPU. It can be easily accelerated usingGPUs. The model was implemented using the low-level interface of the NeuralPDE.jl library which contains the necessarymethods for the generation of the training datasets and of theloss functions starting from the explicit form of the equations and the boundary conditions.

 - Development: old but working code
 - Examples: the results of some idealized and simplified cases
 - Final code: the final working version
 - Saved NN parameters: useful for reproduce results without having to re-learn
