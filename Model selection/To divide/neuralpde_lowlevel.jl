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

@parameters t, x, y, θ
@variables u(..)
@derivatives Dt'~t
@derivatives Dx'~x
@derivatives Dy'~y
@derivatives Dxx''~x
@derivatives Dyy''~y

#2D PDE
eq  = Dt(u(t,x,y,θ)) + u(t,x,y,θ)*Dx(u(t,x,y,θ)) + u(t,x,y,θ)*Dy(u(t,x,y,θ)) ~ Dxx(u(t,x,y,θ)) + Dyy(u(t,x,y,θ))

# Initial and boundary conditions
bcs = [u(0,x,y,θ) ~ exp(x+y)*cos(x+y) ,
       u(2,x,y,θ) ~ exp(x+y)*cos(x+y+4*2) ,
       u(t,0,y,θ) ~ exp(y)*cos(y+4t),
       u(t,2,y,θ) ~ exp(2+y)*cos(2+y+4t) ,
       u(t,x,0,θ) ~ exp(x)*cos(x+4t),
       u(t,x,2,θ) ~ exp(x+2)*cos(x+2+4t)]

# Space and time domains
domains = [x ∈ IntervalDomain(0.0,2.0),
           y ∈ IntervalDomain(0.0,2.0),
           t ∈ IntervalDomain(0.0,2.0)]# Discretization

dx = 0.125; dy= 0.125; dt = 0.125

# Neural network
chain = FastChain(FastDense(3,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

#strategy = GridTraining()
strategy = StochasticTraining()
discretization = PhysicsInformedNN([dt,dx,dy],chain,strategy=strategy)

indvars = [t,x,y]
depvars = [u]
dim = length(domains)

expr_pde_loss_function = build_loss_function(eq,indvars,depvars)
expr_bc_loss_functions = [build_loss_function(bc,indvars,depvars) for bc in bcs]
train_sets = generate_training_sets(domains,dx,bcs,indvars,depvars)

train_domain_set, train_bound_set, train_set = train_sets

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

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

f = NeuralPDE.OptimizationFunction(loss_function, GalacticOptim.AutoZygote())

prob = GalacticOptim.OptimizationProblem(f, initθ)

# optimizer
opt = GalacticOptim.ADAM(0.01)
res = GalacticOptim.solve(prob, opt; cb = cb, maxiters=30)
prob
phi = discretization.phi

ts,xs,ys = [domain.domain.lower:dx:domain.domain.upper for domain in domains]
#u_predict_contourf = reshape([first(phi([t,x,y],res.minimizer)) for t in ts for x in xs for y in ys] ,length(ys),length(xs),length(ts))
#Plots.plot(ts, xs, ys, u_predict_contourf, linetype=:contourf,title = "predict")

#u_predict = [[first(phi([t,x,y],res.minimizer)) for x in xs for y in ys] for t in ts ]
u_predict = [reshape([first(phi([t,x,y],res.minimizer)) for x in xs for y in ys], (length(xs),length(ys))) for t in ts ]
p1= Plots.plot(xs, u_predict[2],title = "t = 0.1");
p2= Plots.plot(xs, u_predict[6],title = "t = 0.5");
p3= Plots.plot(xs, u_predict[end],title = "t = 1");
Plots.plot(p1,p2,p3)
