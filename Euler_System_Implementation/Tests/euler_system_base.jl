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

# Short description of this file: A basic implementation of Euler System

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux, CUDA
using Plots, PyPlot


@parameters x y η t θ
@variables u1(..), u2(..), u3(..), u4(..), u5(..), u6(..), u7(..)
@derivatives Dt'~t
@derivatives Dx'~x
@derivatives Dy'~y
@derivatives Dη'~η
@derivatives Dxx''~x
@derivatives Dyy''~y


#functions
u  = u1(x,y,η,t,θ)
v  = u2(x,y,η,t,θ)
w  = u3(x,y,η,t,θ)
ω  = u4(x,y,η,t,θ)
θm = u5(x,y,η,t,θ)
qm = u6(x,y,η,t,θ)
z  = u7(x,y,η,t,θ)

#physical constants
g  = 9.81 #constant
γ  = 1.4 #air cp/cv=1.4
ρd = 1 #dry air density
p0 = 100000 #reference pressure 10^5 Pa
Rd = 8.31 #gas constant for dry air
ηc = 0.2 #page 8 of Advanced Research WRF V4 (http://dx.doi.org/10.5065/1dfh-6p97)

#simulation inputs
sum_of_qs = 0.1 #read from file
#dx =
#dy =
#dz =
Δx = 1 #dummy
Δy = 1 #dummy

#scale factors
dist = x^2 + y^2 #dummy for distance on earth
Dxd  = 2*x #Dx(dist)
Dyd  = 2*y #Dy(dist)
mx   = Δx/dist
my   = Δy/dist

#definitions
αd = 1/ρd
p  = p0*(Rd*θm/(p0*αd))^γ
c1 = 2*ηc*ηc/((1 - ηc)*(1 - ηc)*(1 - ηc))
c2 = -ηc*(4 + ηc + ηc*ηc)/((1 - ηc)*(1 - ηc)*(1 - ηc))
c3 = 2*(1 + ηc + ηc*ηc)/((1 - ηc)*(1 - ηc)*(1 - ηc))
c4 = -(1 + ηc)/((1 - ηc)*(1 - ηc)*(1 - ηc))
B  = c1 + c2*η + c3*η*η + c4*η*η*η
ps = 10 #dummy
pt = 1 #dummy
pd = B*(ps - pt) + (η - B)*(p0 - pt) + pt
μd = (ps - pt)*(c2 + 2*c3*η + 3*c4*η*η) + (p0 - pt)*(1 - c2 - 2*c3*η - 3*c4*η*η) #[μd = Dη(pd)]
D2pd = (ps - pt)*(2*c3 + 6*c4*η) + (p0 - pt)*(-2*c3 - 6*c4*η) # (maybe find η as a function of pd)
#ω  = Dt(η)
U  = μd*u/my
V  = μd*v/mx
W  = μd*w/my
Ω  = μd*ω/my
Θm = μd*θm
Qm = μd*qm
ϕ  = g*z
α  = αd/(1 + sum_of_qs)


#divergences
#divVu  = Dx(U*u)  + Dy(V*u)  + Dη(Ω*u)
#divVv  = Dx(U*v)  + Dy(V*v)  + Dη(Ω*v)
#divVw  = Dx(U*w)  + Dy(V*w)  + Dη(Ω*w)
#divVθm = Dx(U*θm) + Dy(V*θm) + Dη(Ω*θm)
#divV   = Dx(V)    + Dy(V)    + Dη(V)
#divVqm = Dx(U*qm) + Dy(V*qm) + Dη(Ω*qm)
divVu1 = μd*u/Δy*(u*Dxd + 2*dist*Dx(u))
divVu2 = μd/Δx*(u*v*Dyd + dist*u*Dy(v) + dist*v*Dy(u))
divVu3 = dist/Δy*(D2pd*u*ω + μd*Dη(u)*ω + μd*u*Dη(ω))
divVu  = divVu1 + divVu2 + divVu3

divVv1 = μd/Δy*(u*v*Dxd + dist*u*Dx(v) + dist*v*Dx(u))
divVv2 = μd/Δx*(v*v*Dyd + dist*2*v*Dy(v))
divVv3 = 1/my*(D2pd*w*v + μd*Dη(ω)*v + μd*ω*Dη(v))
divVv  = divVv1 + divVv2 + divVv3

divVw1 = μd/Δy*(Dxd*u*w + dist*Dx(u)*w + dist*u*Dx(w))
divVw2 = μd/Δx*(Dyd*v*w + dist*Dy(v)*w + dist*v*Dy(w))
divVw3 = 1/my*(D2pd*ω*w + μd*Dη(ω)*w   + μd*ω*Dη(w))
divVw  = divVw1 + divVw2 + divVw3

divVθm1 = μd/Δy*(Dxd*u*θm + dist*Dx(u)*θm + dist*u*Dx(θm))
divVθm2 = μd/Δx*(Dyd*v*θm + dist*Dy(v)*θm + dist*v*Dy(θm))
divVθm3 = 1/my*(D2pd*ω*θm + μd*Dη(ω)*θm   + μd*ω*Dη(θm))
divVθm  = divVθm1 + divVθm2 + divVθm3

divV1 = μd/Δx*(Dxd*v + dist*Dx(v))
divV2 = μd/Δx*(Dyd*v + dist*Dy(v))
divV3 = 1/mx*(D2pd*v + μd*Dη(v))
divV  = divV1 + divV2 + divV3

divVqm1 = μd/Δy*(Dxd*u*qm + dist*Dx(u)*qm + dist*u*Dx(qm))
divVqm2 = μd/Δx*(Dyd*v*qm + dist*Dy(v)*qm + dist*v*Dy(qm))
divVqm3 = 1/my*(D2pd*ω*qm + μd*Dη(ω)*qm   + μd*ω*Dη(qm))
divVqm  = divVqm1 +divVqm2 + divVqm3

#inner products
innerV∇ϕ = U*g*Dx(z) + V*g*Dy(z) + Ω*g*Dη(z) #U*Dx(ϕ) + V*Dy(ϕ) + Ω*Dη(ϕ)


#geographic parametrization
Ωe = 1 #dummy #angular rotation rate of earth
ψ  = 1 #dummy #latitude on earth
re = 1000 #dummy #radius of earth
αr = 1 #dummy #local rotation angle between y-axis and the meridians
f  = 2*Ωe*sin(ψ)
e  = 2*Ωe*cos(ψ)

#external forcings
Dymx = -Δx/(dist*dist)*Dyd
Dxmy = -Δy/(dist*dist)*Dxd

FU  =  (mx/my)*(f + u*(my/mx)*Dymx - v*Dxmy)*V - (u/re + e*cos(αr))*W
FV  = -(my/mx)*((f + u*(my/mx)*Dymx - v*Dxmy*U + (v/re - e*sin(αr))*W))
FW  = e*(U*cos(αr) - (mx/my)*V*sin(αr) + 1/re*(u*U + (mx/my)*v*V))
FΘm = 1 #dummy
FQm = 1 #dummy


#equations (Flux-Form Euler)
#eq1 = Dt(U) + divVu + μd*α*Dx(p) + (α/αd)*Dη(p)*Dx(ϕ) ~ FU
#eq2 = Dt(V) + divVv + μd*α*Dy(p) + (α/αd)*Dη(p)*Dy(ϕ) ~ FV
#eq3 = Dt(W) + divVw - g*((α/αd)*Dη(p) - μd)           ~ FW
#eq4 = Dt(Θm) + divVθm                                 ~ FΘm
#eq5 = Dt(μd) + divV                                   ~ 0
#eq6 = Dt(ϕ)  + (1/μd)*(innerV∇ϕ - g*W)                ~ 0
#eq7 = Dt(Qm) + divVqm                                 ~ FQm

eq1 = μd/my*Dt(u) + divVu + μd*α*p0*(Rd/(p0*αd))^γ*γ*θm^(γ-1)*Dx(θm) + (α/αd)*p0*(Rd/(p0*αd))^γ*γ*θm^(γ-1)*Dη(θm)*g*Dx(z) ~ FU
eq2 = μd/mx*Dt(v) + divVv + μd*α*p0*(Rd/(p0*αd))^γ*γ*θm^(γ-1)*Dy(θm) + (α/αd)*p0*(Rd/(p0*αd))^γ*γ*θm^(γ-1)*Dη(θm)*g*Dy(z) ~ FV
eq3 = μd/my*Dt(w) + divVw - g*((α/αd)*p0*(Rd/(p0*αd))^γ*γ*θm^(γ-1)*Dη(θm) - μd)                                           ~ FW


eq4 = μd*Dt(θm) + divVθm                 ~ FΘm #coupling with SFIRE
eq5 = divV                               ~ 0.
eq6 = g*Dt(z)  + (1.0/μd)*(innerV∇ϕ - g*W) ~ 0.
eq7 = μd*Dt(qm) + divVqm                 ~ FQm  #coupling with SFIRE

#system of equations
eqs = [eq1,eq2,eq3,eq4,eq5,eq6,eq7]

# Initial and boundary conditions
bcs = [u1(x,y,η,0,θ) ~ (x^2 + y^2 + η^2)
       u2(x,y,η,0,θ) ~ (x^2 + y^2 + η^2)
       u3(x,y,η,0,θ) ~ (x^2 + y^2 + η^2)
       u4(x,y,η,0,θ) ~ (x^2 + y^2 + η^2)
       u5(x,y,η,0,θ) ~ (x^2 + y^2 + η^2)
       u6(x,y,η,0,θ) ~ (x^2 + y^2 + η^2)
       u7(x,y,η,0,θ) ~ (x^2 + y^2 + η^2)]


# Space and time domains
domains = [x ∈ IntervalDomain(0.0,2.0),
           y ∈ IntervalDomain(0.0,2.0),
           η ∈ IntervalDomain(0.0,2.0),
           t ∈ IntervalDomain(0.0,2.0)]# Discretization

# Discretization
dx = 0.25; dy= 0.25; dη = 0.25; dt = 0.25
# Neural network
input_ = length(domains)
output = length(eqs)
chain = FastChain(FastDense(input_,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,output)) |>gpu

discretization = NeuralPDE.PhysicsInformedNN([dx,dy,dη,dt],
                                             chain,
                                             strategy = NeuralPDE.GridTraining())
pde_system = PDESystem(eqs,bcs,domains,[x,y,η,t],[u1,u2,u3,u4,u5,u6,u7])
prob = NeuralPDE.discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

res = GalacticOptim.solve(prob, ADAM(0.1), progress = true; cb = cb, maxiters=2)
phi = discretization.phi

xs,ys,ηs,ts = [domain.domain.lower:dx:domain.domain.upper for (dx,domain) in zip([dx,dy,dη,dt],domains)]

u_predict = [reshape([first(phi([x,y,η,t],res.minimizer)) for x in xs  for y in ys], (length(xs),length(ys)))  for t in ts ]

#@test u_predict ≈ u_real atol = 200.0

# p1 =plot(xs, ys, u_predict, st=:surface);
# plot(p1)
