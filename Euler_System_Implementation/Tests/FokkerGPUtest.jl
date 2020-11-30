#___  ____       ___ _____   _   _       _ _
#|  \/  | |     |_  /  __ \ | | | |     (_| |
#| .  . | |       | | /  \/ | | | |_ __  _| |_ ___
#| |\/| | |       | | |     | | | | '_ \| | __/ _ \
#| |  | | |___/\__/ | \__/\ | |_| | | | | | || (_) |
#_______\_____\____/ \____/  _____|_|___|_____\_____ _____ _____
#| ___ \        (_)         | | \ \ / / / __  |  _  / __  |  _  |
#| |_/ _ __ ___  _  ___  ___| |_ \ V /  `' / /| |/' `' / /| |/' |
#|  __| '__/ _ \| |/ _ \/ __| __|/   \    / / |  /| | / / |  /| |
#| |  | | | (_) | |  __| (__| |_/ /^\ \ ./ /__\ |_/ ./ /__\ |_/ /
#\_|  |_|  \___/| |\___|\___|\__\/   \/ \_____/\___/\_____/\___/
#              _/ |
#             |__/
#
# This code is part of the proposal of the team "MLJC UniTo" - University of Turin
# for "ProjectX 2020" Climate Change for AI.
# The code is licensed under MIT 3.0
# Please read readme or comments for credits and further information.

# Compiler: Julia 1.5


# update NeuralPDE#quadrature_training
using NeuralPDE
using Quadrature, Cubature, Cuba
using Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Plots
using PyPlot
using CUDA
using CUDAdrv

print("Precompiling Done")


# the example is taken from this article https://arxiv.org/abs/1910.10503
@parameters x θ
@variables p(..)
@derivatives Dx'~x
@derivatives Dxx''~x

#2D PDE
α = 0.3
β = 0.5
_σ = 0.5
# Discretization
dx = 0.05
# here we use normalization condition: dx*p(x,θ) ~ 1 in order to get a non-zero solution.
eq  = [(α - 3*β*x^2)*p(x,θ) + (α*x - β*x^3)*Dx(p(x,θ)) ~ (_σ^2/2)*Dxx(p(x,θ)),
       dx*p(x,θ) ~ 1.]

# Initial and boundary conditions
bcs = [p(-2.2,θ) ~ 0. ,p(2.2,θ) ~ 0. , p(-2.2,θ) ~ p(2.2,θ)]

# Space and time domains
domains = [x ∈ IntervalDomain(-2.2,2.2)]

# Neural network
chain = Chain(Dense(1,12,Flux.σ),Dense(12,12,Flux.σ),Dense(12,1)) |>gpu

discretization = NeuralPDE.PhysicsInformedNN(dx,
                                             chain,
                                             strategy= NeuralPDE.GridTraining()) |>gpu

pde_system = PDESystem(eq,bcs,domains,[x],[p])
prob = NeuralPDE.discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end
CUDA.allowscalar(false)
res = GalacticOptim.solve(prob, ADAM(0.1); cb = cb, maxiters=8000)
phi = discretization.phi

analytic_sol_func(x) = 28.022*exp((1/(2*_σ^2))*(2*α*x^2 - β*x^4))

xs = [domain.domain.lower:dx:domain.domain.upper for domain in domains][1]
u_real  = [analytic_sol_func(x) for x in xs]
u_predict  = [first(phi(x,res.minimizer)) for x in xs]

plot(xs ,u_real, label = "analytic")
plot!(xs ,u_predict, label = "predict")
