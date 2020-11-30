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

# Short description of this file: Non Linear Burger Equation

using NeuralPDE, Flux, ModelingToolkit, Optim, DiffEqFlux
using GalacticOptim
using Base
using Plots, PyPlot
#GalacticOptim
# 3D PDE
@parameters x t θ
@variables u(..)
@derivatives Dxx''~x
#@derivatives Dyy''~y
@derivatives Dx'~x
#@derivatives Dy'~y
@derivatives Dt'~t

Re = 2500.0

# 3D PDE
eq  = Dt(u(x,t,θ)) + u(x,t,θ) * Dx(u(x,t,θ)) ~ 1/Re * Dxx(u(x,t,θ))
#eq  = Dt(u(x,y,t,θ)) + u(x,y,t,θ) * Dx(u(x,y,t,θ)) ~ 1/Re * (Dxx(u(x,y,t,θ)) + Dyy(u(x,y,t,θ)))

# Initial and boundary conditions

A0   = 1.0
muX0 = 0.
muY0 = 0.
sgX0 = 1.0
sgY0 = 1.0

bcs = [u(x,0,θ) ~ A0 * exp(-(((x-muX0)^2/(2*sgX0^2))))]#+((y-muY0)^2/(2*sgY0^2))))]

#bcs = [u(x,y,0,θ) ~ exp(x+y)*cos(x+y) ,
#       u(0,y,t,θ) ~ exp(y)*cos(y+4t),
#       u(2,y,t,θ) ~ exp(2+y)*cos(2+y+4t) ,
#       u(x,0,t,θ) ~ exp(x)*cos(x+4t),
#       u(x,2,t,θ) ~ exp(x+2)*cos(x+2+4t)]

       # Space and time domains
domains = [x ∈ IntervalDomain(-5.0,5.0),
           t ∈ IntervalDomain(-5.0,5.0)]

# Discretization
dx = 0.5; dt = 0.5

# Neural network
chain = FastChain(FastDense(2,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

discretization = NeuralPDE.PhysicsInformedNN([dx,dt],
                                             chain,
                                             strategy = NeuralPDE.StochasticTraining(include_frac=0.9))
pde_system = PDESystem(eq,bcs,domains,[x,t],[u])
prob = NeuralPDE.discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

res = GalacticOptim.solve(prob, ADAM(0.1), progress = true; cb = cb, maxiters=500)
phi = discretization.phi

xs,ts = [domain.domain.lower:dx:domain.domain.upper for domain in domains]

#xs,ys,ts = [domain.domain.lower:dx:domain.domain.upper for domain in domains]

#u_predict = [reshape([first(phi([x,y,t],res.minimizer)) for x in xs for y in ys], (length(xs),length(ys))) for t in ts]


analytic_sol_func(x,t) = [exp(-t)*sin(pi*x), exp(-t)*cos(pi*x), (1+pi^2)*exp(-t)]
#u_real  = [reshape([first(analytic_sol_func(x,t)[i] for x in xs for y in ys], (length(xs),length(ys))) for t in ts]
u_real  = [[analytic_sol_func(x,t)[i] for t in ts for x in xs] for i in 1:3]

u_predict  = [[phi([x,t],res.minimizer)[i] for t in ts for x in xs] for i in 1:1]
u_predict = u_predict[1]
#u_predict = [reshape([first(phi([x,t],res.minimizer)) for x in xs for t in ts])]

diff_u = [abs.(u_real[i] .- u_predict[i] ) for i in 1:3]

p2 = Plots.plot(xs, ts, u_predict, st=:surface,title = "predict")
Plots.plot(p2)

for i in 1:3
    p1 = Plots.plot(xs, ts, u_real[i], st=:surface,title = "u$i, analytic");
    p2 = Plots.plot(xs, ts, u_predict[i], st=:surface,title = "predict");
    #p3 = Plots.plot(xs, ts, diff_u[i],linetype=:contourf,title = "error");
    #plot(p1,p2,p3)
    Plots.plot(p2)
    Plots.savefig("sol_U$i")
end



maxlim = maximum(maximum(u_predict[t]) for t = 1:length(ts))
minlim = minimum(minimum(u_predict[t]) for t = 1:length(ts))


result = @animate for time = 1:length(ts)
    Plots.plot(xs, ys, u_predict[time],st=:surface,camera=(30,30), zlim=(minlim,maxlim), clim=(minlim,maxlim),
                title = string("ψ: max = ",round(maxlim, digits = 3)," min = ", round(minlim, digits = 3),"\\n"," t = ",time))
end

gif(result, "burger_surface.gif", fps = 6)

result = @animate for time = 1:length(ts)
    Plots.plot(xs, ys, u_predict[time],st=:contour,camera=(30,30), zlim=(minlim,maxlim), clim=(minlim,maxlim),
                title = string("ψ: max = ",round(maxlim, digits = 3)," min = ", round(minlim, digits = 3),"\\n"," t = ",time))
end

gif(result, "burger_contour.gif", fps = 6)
