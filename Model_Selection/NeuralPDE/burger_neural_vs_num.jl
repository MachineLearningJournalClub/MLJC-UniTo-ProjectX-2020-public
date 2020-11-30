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

# Short description of this file: Burger's equation Neural vs Numerical
#Implementation

using ModelingToolkit, NeuralPDE, DiffEqFlux, Flux, Plots, GalacticOptim, Test, Optim

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

@parameters t, x, θ
@variables u(..)
@derivatives Dt'~t
@derivatives Dx'~x
@derivatives Dxx''~x

#2D PDE
eq  = Dt(u(t,x,θ)) + u(t,x,θ)*Dx(u(t,x,θ)) - (0.01/pi)*Dxx(u(t,x,θ)) ~ 0

# Initial and boundary conditions
bcs = [u(x,0,θ) ~ 2*x]
#u(0,x,θ) ~ -sin(pi*x),
       #u(t,-1,θ) ~ 0.,
       #u(t,1,θ) ~ 0.]

# Space and time domains
domains = [t ∈ IntervalDomain(0.0,10.0),
           x ∈ IntervalDomain(-1.0,1.0)]
# Discretization
dx = 0.1

# Neural network
chain = FastChain(FastDense(2,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

strategy = GridTraining()
discretization = PhysicsInformedNN(dx,chain,strategy=strategy)

indvars = [t,x]
depvars = [u]
dim = length(domains)

expr_pde_loss_function = build_loss_function(eq,indvars,depvars)
expr_bc_loss_functions = [build_loss_function(bc,indvars,depvars) for bc in bcs]

train_sets = generate_training_sets(domains,dx,bcs,indvars,depvars)

train_domain_set, train_bound_set, train_set= train_sets


phi = discretization.phi
autodiff = discretization.autodiff
derivative = discretization.derivative

initθ = discretization.initθ

pde_loss_function = get_loss_function(eval(expr_pde_loss_function),
                                      train_domain_set,
                                      phi,
                                      derivative,
                                      strategy)

bc_loss_function = get_loss_function(eval.(expr_bc_loss_functions),
                                     train_bound_set,
                                     phi,
                                     derivative,
                                     strategy)

function loss_function(θ,p)
    return pde_loss_function(θ) + bc_loss_function(θ)
end

f = OptimizationFunction(loss_function, GalacticOptim.AutoZygote())

prob = GalacticOptim.OptimizationProblem(f, initθ)

# optimizer
opt = Optim.BFGS()

res = GalacticOptim.solve(prob, opt; cb = cb, maxiters=15)

phi = discretization.phi

ts,xs = [domain.domain.lower:dx:domain.domain.upper for domain in domains]
u_predict_contourf = reshape([first(phi([t,x],res.minimizer)) for t in ts for x in xs] ,length(xs),length(ts))
plot(ts, xs, u_predict_contourf, linetype=:surface,title = "predict")

u_predict = [[first(phi([t,x],res.minimizer)) for x in xs] for t in ts ]
p1= plot(xs, u_predict[2],title = "t = 0.1");
p2= plot(xs, u_predict[6],title = "t = 0.5");
p3= plot(xs, u_predict[end],title = "t = 1");
plot(p1,p2,p3)

analytic_sol_func(x,t) = 2*x/(1+2*t)
#analytic_sol_func(x,t) = [exp(-t)*sin(pi*x), exp(-t)*cos(pi*x), (1+pi^2)*exp(-t)]
#u_real  = [[analytic_sol_func(x,t)[i] for t in ts for x in xs] for i in 1:3]
u_real  = [analytic_sol_func(x,t) for t in ts for x in xs]

#=
for i in 1:3
    p1 = Plots.plot(xs, ts, u_real[i], st=:surface,title = "u$i, analytic");
    Plots.plot(p1)
    Plots.savefig("sol_real$i")
end
=#

plot(ts, xs, u_real, linetype=:surface,title = "real_contour")
