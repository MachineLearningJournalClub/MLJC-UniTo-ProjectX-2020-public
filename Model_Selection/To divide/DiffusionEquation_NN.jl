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

# Short description of this file: 2D REACTION DIFFUSION EQUATION SOLVED AS A
#STANDARD ODEPROBLEM WITH A NN

#Warning: Reset Julia Kernel (pre-compilation required) to avoid errors in
#redefinitions of constants. 

import Pkg; Pkg.add("OrdinaryDiffEq"); Pkg.add("RecursiveArrayTools"); Pkg.add("DiffEqOperators"); Pkg.add("CuArrays")
import Pkg; Pkg.add("PyPlot")
using PyPlot
using OrdinaryDiffEq, RecursiveArrayTools, LinearAlgebra,
      DiffEqOperators, Flux

# Define the constants for the PDE
const α₂ = 1.0f0
const α₃ = 1.0f0
const β₁ = 1.0f0
const β₂ = 1.0f0
const β₃ = 1.0f0
const r₁ = 1.0f0
const r₂ = 1.0f0
const D = 100.0f0
const γ₁ = 0.1f0
const γ₂ = 0.1f0
const γ₃ = 0.1f0
const N = 100
const X = reshape([i for i in 1:N for j in 1:N],N,N)
const Y = reshape([j for i in 1:N for j in 1:N],N,N)
const α₁ = 1.0f0.*(X.>=80)

const Mx = Array(Tridiagonal([1.0f0 for i in 1:N-1],[-2.0f0 for i in 1:N],[1.0f0 for i in 1:N-1]))
const My = copy(Mx)
Mx[2,1] = 2.0
Mx[end-1,end] = 2.0
My[1,2] = 2.0
My[end,end-1] = 2.0

# Define the initial condition as normal arrays
u0 = rand(Float32,N,N,3)
const MyA = zeros(Float32,N,N)
const AMx = zeros(Float32,N,N)
const DA = zeros(Float32,N,N)

# Define the discretized PDE as an ODE function
function f(_du,_u,p,t)
  u = reshape(_u,N,N,3)
  du= reshape(_du,N,N,3)
  A = @view u[:,:,1]
  B = @view u[:,:,2]
  C = @view u[:,:,3]
  dA = @view du[:,:,1]
  dB = @view du[:,:,2]
  dC = @view du[:,:,3]
  mul!(MyA,My,A)
  mul!(AMx,A,Mx)
  @. DA = D*(MyA + AMx)
  @. dA = DA + α₁ - β₁*A - r₁*A*B + r₂*C
  @. dB = α₂ - β₂*B - r₁*A*B + r₂*C
  @. dC = α₃ - β₃*C + r₁*A*B - r₂*C
end

# Solve the ODE
prob = ODEProblem(f,vec(u0),(0.0f0,100.0f0))
@time sol = solve(prob,BS3(),  progress=true,saveat = 5.0)
@time sol = solve(prob,ROCK2(),progress=true,saveat = 5.0)


using Plots; pyplot()
p1 = surface(X,Y,reshape(sol[end],N,N,3)[:,:,1],title = "[A]")
p2 = surface(X,Y,reshape(sol[end],N,N,3)[:,:,2],title = "[B]")
p3 = surface(X,Y,reshape(sol[end],N,N,3)[:,:,3],title = "[C]")
plot(p1,p2,p3,layout=grid(3,1))
savefig("neural_pde_training_data.png")

using DiffEqFlux, Flux

u0 = param(u0)
tspan = (0.0f0,100.0f0)

ann = Chain(Dense(3,50,tanh), Dense(50,3))
p1 = DiffEqFlux.destructure(ann)
ps = Flux.params(ann)

_ann = (u,p) -> reshape(p[3*50+51 : 2*3*50+50],3,50)*
                    tanh.(reshape(p[1:3*50],50,3)*u + p[3*50+1:3*50+50]) + p[2*3*50+51:end]

function dudt_(_u,p,t)
  u = reshape(_u,N,N,3)
  A = u[:,:,1]
  DA = D .* (A*Mx + My*A)
  _du = mapslices(x -> _ann(x,p),u,dims=3)
  du = reshape(_du,N,N,3)
  x = vec(cat(du[:,:,1]+DA,du[:,:,2],du[:,:,3],dims=3))
end

prob = ODEProblem(dudt_,vec(Flux.data(u0)),tspan,Flux.data(p1))
@time diffeq_fd(p1,Array,length(u0)*length(0.0f0:5.0f0:100.0f0),prob,ROCK2(),progress=true,
                saveat=0.0f0:5.0f0:100.0f0)

function predict_fd()
  diffeq_fd(p1,Array,length(u0)*length(0.0f0:5.0f0:100.0f0),prob,ROCK2(),progress=true,
                  saveat=0.0f0:5.0f0:100.0f0)
end

function loss_fd()
  _sol = predict_fd()
  loss = sum(abs2,Array(sol) .- _sol)
  @show loss
  loss
end
loss_fd()

data = Iterators.repeated((), 10)
opt = ADAM(0.025)

Flux.train!(loss_fd, ps, data, opt)
