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

# Short description of this file: Level Set Implementation First Stage

# Reference paper: https://gmd.copernicus.org/articles/4/591/2011/gmd-4-591-2011.pdf

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux, BenchmarkTools
using Plots #GalacticOptim
using PyPlot

@parameters t x y θ
@variables u(..) #u corresponds to ψ in reference paper, z is the terrain height
@derivatives Dt'~t
@derivatives Dx'~x
@derivatives Dy'~y
@derivatives Dxx''~x
@derivatives Dyy''~y

#operators
z    = x^2 + y^2
gn   = (Dx(u(t,x,y,θ))^2 + Dy(u(t,x,y,θ))^2)^(1/2) #gradient norm
∇u   = [Dx(u(t,x,y,θ)), Dy(u(t,x,y,θ))]
#Δu   = Dxx(u(t,x,y,θ)) + Dyy(u(t,x,y,θ))
Dxz  = 2*x
Dyz  = 2*y
#∇z   = [Dx(z),Dy(z)]

#fuel dependent coefficients
n    = ∇u/gn #normal to the fire region
tanϕ = sum([Dxz,Dyz].*n)
#f = open("namelist.fire.txt")

#Fuel parameters given by file (convert to Imperial unit system!!!!)
windrf = [0.36, 0.36, 0.44,  0.55,  0.42,  0.44,  0.44, 0.36, 0.36, 0.36,  0.36,  0.43,  0.46, 1e-7]
fgi    = [0.166, 0.897, 1.076, 2.468, 0.785, 1.345, 1.092, 1.121, 0.780, 2.694, 2.582, 7.749, 13.024, 1.e-7]
fueldepthm = [0.305, 0.305, 0.762, 1.829, 0.61,  0.762, 0.762, 0.061, 0.061, 0.305, 0.305, 0.701, 0.914, 0.305]
savr = [3500., 2784., 1500., 1739., 1683., 1564., 1562., 1889., 2484., 1764., 1182., 1145., 1159., 3500.]
fuelmce = [0.12, 0.15, 0.25, 0.20, 0.20, 0.25, 0.40, 0.30, 0.25, 0.25, 0.15, 0.20, 0.25, 0.12]
fueldens = [32.,32.,32.,32.,32.,32.,32.,32.,32.,32.,32.,32.,32.,32.] #from namelist.fire: "! 32 if solid, 19 if rotten"
st = [0.0555, 0.0555, 0.0555, 0.0555, 0.0555, 0.0555, 0.0555, 0.0555, 0.0555, 0.0555, 0.0555, 0.0555, 0.0555, 0.0555]
se = [0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010]
cmbcnst  = 17.433e+06
fuelmc_g = 0.08  #or 0.09
weight = [7.,  7.,  7., 180., 100., 100., 100., 900., 900., 900., 900., 900., 900., 7.]


a     = windrf[1]
zf    =
z0    =
w     = weight[1]
wl    = fgi[1]
δm    = fueldepthm[1]
sigma = savr[1]
Mx    = fuelmce[1]
ρP    = fueldens[1]
ST    = st[1]
SE    = se[1]
h     = cmbcnst[1]
Mf    = fuelmc_g[1]

#Fire spread rate equations
βop  = 0.1 #dummy
U    = 2 #wind vector (dummy)
w0   = wl/(1 + Mf)
ρb   = w0/δm #different from paper for units reasons
β    = ρb/ρP
ξ    = exp((0.792 + 0.618*sigma^0.5)*(β+0.1))
ηs   = 0.174*SE^-0.19 #probably mistaken in the paper
ηM   = 1 - 2.59*Mf/Mx + 5.11*(Mf/Mx)^2 - 3.52*(Mf/Mx)^3
wn   = w0/(1 + ST)
Γmax = sigma^(1.5)/(495 + 0.594*sigma^(1.5))
A    = 1/(4.77*sigma^(0.1) - 7.27)
Γ    = Γmax*(β/βop)^A*exp(A*(1 - β/βop))
ϵ    = exp(-138/sigma)
Qig  = 250*β + 1116*Mf
C    = 7.47*exp(-0.133*sigma^0.55)
Ua   = a*U #U is the wind vector
E    = 0.715*exp(-0.000359*sigma)
IR   = Γ*wn*h*ηM*ηs
R0   = IR*ξ/(ρb*ϵ*Qig)             #spread rate without wind
ϕw   = C*max(Ua^β, (β/βop)^E)      #wind factor
ϕS   = 5.275*β^-0.3*tanϕ^2       #slope factor (problem is we need to give tanϕ as a number)

S = 2 #R0*(1 + ϕw + ϕS) #fire spread rate [max(S0, R0 + c*min(e,max(0,U))^b + d*max(0,tanϕ)^2)]

#level set equation
eq = Dt(u(t,x,y,θ)) + S*gn ~ 0
#eq2 = z(x,y) - x^2 - y^2 ~ 0

#eqs = [eq1,eq2]

# Space and time domains
domains = [x ∈ IntervalDomain(0.0,2.0),
           y ∈ IntervalDomain(0.0,2.0),
           t ∈ IntervalDomain(0.0,2.0)] # Discretization

dx = 0.0125; dy = 0.0125; dt = 0.0125;

#ts,xs,ys = [domain.domain.lower:dx:domain.domain.upper for domain in domains]

# Initial and boundary conditions
bcs = [u(0,x,y,θ) ~ 0]

# LOW LEVEL
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
train_sets = generate_training_sets(domains,[dt,dx,dy],bcs,indvars,depvars)

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
res = GalacticOptim.solve(prob, opt; cb = cb, maxiters = 30)
phi = discretization.phi

ts,xs,ys = [domain.domain.lower:dx:domain.domain.upper for domain in domains]

#u_predict_contourf = reshape([first(phi([t,x,y],res.minimizer)) for t in ts for x in xs for y in ys] ,length(ys),length(xs),length(ts))
#Plots.plot(ts, xs, ys, u_predict_contourf, linetype=:contourf,title = "predict")

#u_predict = [[first(phi([t,x,y],res.minimizer)) for x in xs for y in ys] for t in ts ]
u_predict = [reshape([first(phi([t,x,y],res.minimizer)) for x in xs for y in ys], (length(xs),length(ys))) for t in ts]
p1= Plots.plot(xs, u_predict[2],title = "t = 0.1");
p2= Plots.plot(xs, u_predict[6],title = "t = 0.5");
p3= Plots.plot(xs, u_predict[end],title = "t = 1");
Plots.plot(p1,p2,p3)




#HIGH LEVEL
chain = FastChain(FastDense(3,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

discretization = NeuralPDE.PhysicsInformedNN([dt,dx,dy],
                                             chain,
                                             strategy = NeuralPDE.StochasticTraining(include_frac=0.9))
pde_system = PDESystem(eq,bcs,domains,[t,x,y],[u])
prob = NeuralPDE.discretize(pde_system,discretization)

res = GalacticOptim.solve(prob, ADAM(0.04), progress = false; cb = cb, maxiters=300)
phi = discretization.phi


xs,ys,ts = [domain.domain.lower:dx:domain.domain.upper for domain in domains]

analytic_sol_func(x,y,t) = []
u_real  = [[analytic_sol_func(t,x)[i] for t in ts for x in xs] for i in 1:3]
u_predict  = [[phi([x,y,t],res.minimizer)[i] for x in xs for y in ys for t in ts] for i in 1:4]
diff_u = [abs.(u_real[i] .- u_predict[i] ) for i in 1:3]

for i in 1:3
    p1 = plot(xs, ts, u_real[i], st=:surface,title = "u$i, analytic");
    p2 = plot(xs, ts, u_predict[i], st=:surface,title = "predict");
    p3 = plot(xs, ts, diff_u[i],linetype=:contourf,title = "error");
    plot(p1,p2,p3)
    savefig("sol_u$i")
end


#Another Trial
@parameters t x y
@variables u(..), X(..), Y(..), z(..) #u corresponds to ψ in reference paper, z is the terrain height
@derivatives Dt'~t
@derivatives Dx'~x
@derivatives Dy'~y

#operators
∇u = [Dt(u(X(t),Y(t)))/Dt(X(t)),Dt(u(X(t),Y(t)))/Dt(Y(t))]
gn = ((Dt(u(X(t),Y(t)))/Dt(X(t)))^2 + (Dt(u(X(t),Y(t)))/Dt(X(t)))^2)^(1/2) #gradient norm
∇z = [Dx(z(x,y)),Dy(z(x,y))]

#fuel dependent coefficients
n    = ∇u/gn #normal to the fire region
S0   = 1
R0   = 3
b    = 2
c    = 1
d    = 1
e    = 1
U    = 4#wind vector
tanϕ = ∇z.*n

#definitions
S = 1#max(S0, R0 + c*min(e,max(0,U))^b + d*max(0,tanϕ)^2)

#level set equation
eq = Dt(u(X(t),Y(t))) + gn ~ 0

# Initial and boundary conditions


# Space and time domains
domains = [x ∈ IntervalDomain(0.0,2.0),
           y ∈ IntervalDomain(0.0,2.0),
           t ∈ IntervalDomain(0.0,2.0)]# Discretization

dx = 0.125; dy= 0.125; dt = 0.125;

# Neural network
chain = FastChain(FastDense(7,32,Flux.σ),FastDense(32,32,Flux.σ),FastDense(32,1))

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
