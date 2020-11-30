![Logo](/Support_Materials/Assets/Logo_MLJC.png)

<h1 align="center">
  Level Set Implementations
</h1>

The level-set is the mathematical core for calculating the spread of the fire. The minimization of the loss functions is the process that actually solves the PDE and constitutes the main load for the CPU. It can be easily accelerated usingGPUs. The model was implemented using the low-level interface of the [NeuralPDE.jl](https://github.com/SciML/NeuralPDE.jl)  library which contains the necessary methods for the generation of the training datasets and of the loss functions starting from the explicit form of the equations and the boundary conditions.

 * [Development](/Level_Set_Implementation/Development/): old but working code;
 * [Examples](/Level_Set_Implementation/Examples/): the results of some idealized and simplified cases;
 * [Final code](/Level_Set_Implementation/Final Code/): the final working version;
 * [Saved NN parameters](/Level_Set_Implementation/Saved NN params/): useful for reproduce results without having to re-learn.
