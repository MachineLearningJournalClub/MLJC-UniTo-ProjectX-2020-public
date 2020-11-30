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

# Compiler: Julia 1.5.3

# Short description of this file: Poisson's 2D Equations with GPU

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Plots
using CUDA

@parameters x y θ
@variables u(..)
@derivatives Dxx''~x
@derivatives Dyy''~y

# 2D PDE
eq  = Dxx(u(x,y,θ)) + Dyy(u(x,y,θ)) ~ sin(2π*x)*cos(2π*y)

# Boundary conditions
bcs = [u(0,y,θ) ~ 0.f0, u(1,y,θ) ~ -sin(pi*1)*sin(pi*y),
       u(x,0,θ) ~ 0.f0, u(x,1,θ) ~ -sin(pi*x)*sin(pi*1)]
# Space and time domains
domains = [x ∈ IntervalDomain(0.0,1.0),
           y ∈ IntervalDomain(0.0,1.0)]

# Neural network
dim = 2 # number of dimensions
chain = Chain(Dense(dim,16,Flux.σ),Dense(16,16,Flux.σ),Dense(16,1))

# Discretization
dx = 0.05
discretization = PhysicsInformedNN(dx,
                                   chain,
                                   strategy = GridTraining())

pde_system = PDESystem(eq,bcs,domains,[x,y],[u])
prob = discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

res = GalacticOptim.solve(prob, BFGS(); cb = cb, maxiters=1000)
phi = discretization.phi

xs,ys = [domain.domain.lower:dx/10:domain.domain.upper for domain in domains]
analytic_sol_func(x,y) = -sin(2π*x)*cos(2π*y)/(8π^2)

u_predict = reshape([first(phi([x,y],res.minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
u_real = reshape([analytic_sol_func(x,y) for x in xs for y in ys], (length(xs),length(ys)))
diff_u = abs.(u_predict .- u_real)


p1 = Plots.plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
p2 = Plots.plot(xs, ys, u_predict, linetype=:contourf,title = "predict");
p3 = Plots.plot(xs, ys, diff_u,linetype=:contourf,title = "error");
plot(p1,p2,p3)

p1 = Plots.plot(xs, ys, u_real, linetype=:surface,title = "analytic");
p2 = Plots.plot(xs, ys, u_predict, linetype=:surface,title = "predict");
p3 = Plots.plot(xs, ys, diff_u,linetype=:surface,title = "error");
plot(p1,p2,p3)
savefig("graph.pdf")
