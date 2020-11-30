using Pkg
Pkg.add("ModelingToolkit")
Pkg.add("DiffEqFlux")
Pkg.add("Flux")
Pkg.add("Plots")
Pkg.add("Test")
Pkg.add("Optim")
Pkg.add("CUDA")

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Plots #GalacticOptim
using PyPlot


@parameters t, x, θ
@variables u1(..), u2(..), u3(..)
@derivatives Dt'~t
@derivatives Dtt''~t
@derivatives Dx'~x
@derivatives Dxx''~x

eqs = [Dtt(u1(t,x,θ)) ~ Dxx(u1(t,x,θ)) + u3(t,x,θ)*sin(pi*x),
       Dtt(u2(t,x,θ)) ~ Dxx(u2(t,x,θ)) + u3(t,x,θ)*cos(pi*x),
       0. ~ u1(t,x,θ)*sin(pi*x) + u2(t,x,θ)*cos(pi*x) - exp(-t)]

bcs = [u1(0,x,θ) ~ sin(pi*x),
       u2(0,x,θ) ~ cos(pi*x),
       Dt(u1(0,x,θ)) ~ -sin(pi*x),
       Dt(u2(0,x,θ)) ~ -cos(pi*x),
       u1(t,0,θ) ~ 0.,
       u2(t,0,θ) ~ exp(-t),
       u1(t,1,θ) ~ 0.,
       u2(t,1,θ) ~ -exp(-t),
       u1(t,0,θ) ~ u1(t,1,θ),
       u2(t,0,θ) ~ -u2(t,1,θ)]

# Space and time domains
domains = [t ∈ IntervalDomain(0.0,1.0),
           x ∈ IntervalDomain(0.0,1.0)]
# Discretization
dx = 0.1
# Neural network
input_ = length(domains)
output = length(eqs)
chain = FastChain(FastDense(input_,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,output))

strategy = GridTraining()
discretization = PhysicsInformedNN(dx,chain,strategy=strategy)

pde_system = PDESystem(eqs,bcs,domains,[t,x],[u1,u2,u3])
prob = discretize(pde_system,discretization)

res = GalacticOptim.solve(prob,Optim.BFGS(); cb = cb, maxiters=5000)
phi = discretization.phi

ts,xs = [domain.domain.lower:dx:domain.domain.upper for domain in domains]

analytic_sol_func(t,x) = [exp(-t)*sin(pi*x), exp(-t)*cos(pi*x), (1+pi^2)*exp(-t)]
u_real  = [[analytic_sol_func(t,x)[i] for t in ts for x in xs] for i in 1:3]
u_predict  = [[phi([t,x],res.minimizer)[i] for t in ts for x in xs] for i in 1:3]
diff_u = [abs.(u_real[i] .- u_predict[i] ) for i in 1:3]

p1 = plot(xs, ts, u_real[1], st=:surface,title = "u1, analytic");
p2 = plot(xs, ts, u_predict[1], st=:surface,title = "predict");
p3 = plot(xs, ts, diff_u[1],linetype=:contourf,title = "error");
plot(p1,p2,p3)

p1 = plot(xs, ts, u_real[2], st=:surface,title = "u2, analytic");
p2 = plot(xs, ts, u_predict[2], st=:surface,title = "predict");
p3 = plot(xs, ts, diff_u[2],linetype=:contourf,title = "error");
plot(p1,p2,p3)

p1 = plot(xs, ts, u_real[3], st=:surface,title = "u, analytic");
p2 = plot(xs, ts, u_predict[3], st=:surface,title = "predict");
p3 = plot(xs, ts, diff_u[3],linetype=:contourf,title = "error");
plot(p1,p2,p3)
