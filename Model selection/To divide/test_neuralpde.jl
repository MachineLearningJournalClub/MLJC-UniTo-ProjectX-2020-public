using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Plots
#GalacticOptim
# 3D PDE
@parameters x y t θ
@variables u(..)
@derivatives Dxx''~x
@derivatives Dyy''~y
@derivatives Dt'~t

# 3D PDE
eq  = Dt(u(x,y,t,θ)) ~ Dxx(u(x,y,t,θ)) + Dyy(u(x,y,t,θ))
# Initial and boundary conditions
bcs = [u(x,y,0,θ) ~ exp(x+y)*cos(x+y),
       u(0,y,t,θ) ~ exp(y)*cos(y+4t),
       u(2,y,t,θ) ~ exp(2+y)*cos(2+y+4t) ,
       u(x,0,t,θ) ~ exp(x)*cos(x+4t),
       u(x,2,t,θ) ~ exp(x+2)*cos(x+2+4t)]
# Space and time domains
domains = [x ∈ IntervalDomain(0.0,2.0),
           y ∈ IntervalDomain(0.0,2.0),
           t ∈ IntervalDomain(0.0,2.0)]

# Discretization
dx = 0.25; dy= 0.25; dt = 0.25
# Neural network
chain = FastChain(FastDense(3,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))


discretization = NeuralPDE.PhysicsInformedNN([dx,dy,dt],
                                             chain,
                                             strategy = NeuralPDE.StochasticTraining(include_frac=0.9))


pde_system = PDESystem(eq,bcs,domains,[x,y,t],[u])
prob = NeuralPDE.discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

res = GalacticOptim.solve(prob, ADAM(0.1), progress = false; cb = cb, maxiters=3000)
phi = discretization.phi


ts,xs = [domain.domain.lower:dx:domain.domain.upper for domain in domains]

analytic_sol_func(t,x) = [exp(-t)*sin(pi*x), exp(-t)*cos(pi*x), (1+pi^2)*exp(-t)]
u_real  = [[analytic_sol_func(t,x)[i] for t in ts for x in xs] for i in 1:3]
                                   
# errore dimensionale nella riga seguente
u_predict  = [[phi([t,x],res.minimizer)[i] for t in ts for x in xs] for i in 1:3]
diff_u = [abs.(u_real[i] .- u_predict[i] ) for i in 1:3]

for i in 1:3
    p1 = plot(xs, ts, u_real[i], st=:surface,title = "u$i, analytic");
    p2 = plot(xs, ts, u_predict[i], st=:surface,title = "predict");
    p3 = plot(xs, ts, diff_u[i],linetype=:contourf,title = "error");
    plot(p1,p2,p3)
    savefig("sol_u$i")
end
