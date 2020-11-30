## Example 2, ## Fokker-Planck equation
# the example took from this article https://arxiv.org/abs/1910.10503
using Flux
println("NNPDE_tests")
using DiffEqFlux
println("Starting Soon!")
using ModelingToolkit
#using DiffEqBase
using Test, NeuralPDE
using GalacticOptim
using Optim
using Plots
using Random

Random.seed!(100)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

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
# here we use normalization condition: dx*p(x,θ) ~ 1, in order to get non-zero solution.
eq  = [(α - 3*β*x^2)*p(x,θ) + (α*x - β*x^3)*Dx(p(x,θ)) ~ (_σ^2/2)*Dxx(p(x,θ)),
       dx*p(x,θ) ~ 1.]
# Initial and boundary conditions
bcs = [p(-2.2,θ) ~ 0. ,p(2.2,θ) ~ 0. , p(-2.2,θ) ~ p(2.2,θ)]
# Space and time domains
domains = [x ∈ IntervalDomain(-2.2,2.2)]
# Neural network
chain = FastChain(FastDense(1,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1)) #|>gpu
discretization = NeuralPDE.PhysicsInformedNN(dx,
                                             chain,
                                             strategy= NeuralPDE.GridTraining())
pde_system = PDESystem(eq,bcs,domains,[x],[p])

NumDiscretization = MOLFiniteDifference(0.1)
numProb = DiffEqBase.discretize(pde_system, NumDiscretization)
numSol = DifferentialEquations.solve(numProb, RK4(), reltol=1e-5,abstol=1e-5)



prob = NeuralPDE.discretize(pde_system,discretization)
res = GalacticOptim.solve(prob, BFGS(); cb = cb, maxiters=1000)
phi = discretization.phi
analytic_sol_func(x) = 28*exp((1/(2*_σ^2))*(2*α*x^2 - β*x^4))
xs = [domain.domain.lower:dx:domain.domain.upper for domain in domains][1]
u_real  = [analytic_sol_func(x) for x in xs]
u_predict  = [first(phi(x,res.minimizer)) for x in xs]
@test u_predict ≈ u_real atol = 20.0

Plots.plot(xs ,u_real, label = "analytic")
Plots.plot!(xs ,u_predict, label = "predict")
