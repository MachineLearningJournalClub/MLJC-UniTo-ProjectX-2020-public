using NeuralPDE, Flux, ModelingToolkit, Optim, DiffEqFlux
using GalacticOptim
using Base
using SymPy

@parameters x t θ
@variables u(..)
@derivatives Dxx''~x
@derivatives Dt'~t

quasi_dirac = function(x,a)
    return 1/(a*pi^0.5)*exp(-(x/a)*(x/a))
end

quasi_dirac(0,0)

# 3D PDE
eq  = Dt(u(x,t,θ)) ~ Dxx(u(x,t,θ))
# Initial and boundary conditions
bcs = [u(x,0,θ) ~ quasi_dirac(x,1/100000)]
# Space and time domains
domains = [x ∈ IntervalDomain(0.0,2.0),
           t ∈ IntervalDomain(0.0,2.0)]

# Discretization
dx = 0.25; dt = 0.25
# Neural network
chain = FastChain(FastDense(2,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

discretization = NeuralPDE.PhysicsInformedNN([dx,dt],
                                             chain,
                                             strategy = NeuralPDE.GridTraining())
pde_system = PDESystem(eq,bcs,domains,[x,t],[u])
prob = NeuralPDE.discretize(pde_system,discretization)

res = GalacticOptim.solve(prob, ADAM(0.01), progress = false; cb = cb, maxiters=1000)
phi = discretization.phi


ts,xs = [domain.domain.lower:dx:domain.domain.upper for domain in domains]


u_real  = [[analytic_sol_func(t,x)[i] for t in ts for x in xs] for i in 1:3]
u_predict  = [[phi([t,x],res.minimizer)[i] for t in ts for x in xs] for i in 1:3]
diff_u = [abs.(u_real[i] .- u_predict[i] ) for i in 1:3]

for i in 1:3
    p1 = plot(xs, ts, u_real[i], st=:surface,title = "u$i, analytic");
    p2 = plot(xs, ts, u_predict[i], st=:surface,title = "predict");
    p3 = plot(xs, ts, diff_u[i],linetype=:contourf,title = "error");
    plot(p1,p2,p3)
    savefig("sol_u$i")
end
