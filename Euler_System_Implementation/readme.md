![Logo](/Support_Materials/Assets/Logo_MLJC.png)
<h1 align="center">
  Euler System Implementation
</h1>

Our main results regarding the 7-dimensional Euler System are contained in the [Final Code](/Euler_System_Implementation/Final_Code/) folder:

* In [euler_system_7_func.jl](/Euler_System_Implementation/Final_Code/euler_system_7_func.jl) we implemented the system in its original form, i.e. with 4 independent variables and 7 target functions.
* In [euler_system_6-func.jl](/Euler_System_Implementation/Final_Code/euler_system_6-func.jl) we removed one of the seven target function in light of an approximation we had to include, which is keeping the coordinate pressure \eta constant. This caused the function \omega to automatically assume a zero value, since it is defined as a derivative with respect to \eta.
* In [euler_system_6-func_eps.jl](/Euler_System_Implementation/Final_Code/euler_system_6-func_eps.jl) we tried to overcome an instability issue with regularization: we substituted all the functions that appeared in the denominators with max(f,eps), where f is a function.

We also started implementing a numerical solution of the same system, which could serve as a validation of the results obtained with PINNs. After trying different strategies, we settled on the library [_DiffentialEquations.jl_](https://diffeq.sciml.ai/v2.0/), but we need more time to complete this demanding job because of difficulties linked to the extension of matrix calculus to tensors.
