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

# Short description of this file: 2D REACTION DIFFUSION EQUATION SOLVED WITH A CNN

#Warning: Reset Julia Kernel (pre-compilation required) to avoid errors in
#redefinitions of constants

using Pkg#; Pkg.activate("."); Pkg.instantiate()
Pkg.add("BSON")
Pkg.add("DifferentialEquations")

using PyPlot, Printf
using LinearAlgebra
using Flux, DiffEqFlux, Optim, DiffEqSensitivity
using BSON: @save, @load
using Flux: @epochs
using DifferentialEquations


#Domain definition
X = 1.0; T = 5.0;
dx = 0.04; dt = T/10;
Y = 1.0;
dy = 0.04;
x = collect(0:dx:X);
y = collect(0:dy:Y);
t = collect(0:dt:T);
Nx = 128 # Int64(X/dx+1);
Ny = 128 #Int64(Y/dy+1);
Nt = Int64(T/dt+1);

########################
# Generate training data
########################


const α₂ = 1.0
const α₃ = 1.0
const β₁ = 1.0
const β₂ = 1.0
const β₃ = 1.0
const r₁ = 1.0
const r₂ = 1.0
const DD = 100.0
const γ₁ = 0.1
const γ₂ = 0.1
const γ₃ = 0.1
const N = 128
const XX = reshape([i for i in 1:N for j in 1:N],N,N)
const YY = reshape([j for i in 1:N for j in 1:N],N,N)
const aaa = 1.0.*(XX.>=4*N/5)

const Mx = Tridiagonal([1.0 for i in 1:N-1],[-2.0 for i in 1:N],[1.0 for i in 1:N-1])
const My = copy(Mx)
Mx[2,1] = 2.0
Mx[end-1,end] = 2.0
My[1,2] = 2.0
My[end,end-1] = 2.0

# Define the discretized PDE as an ODE function
const MyA = zeros(N,N)
const AMx = zeros(N,N)
const DA = zeros(N,N)

function f(du,u,p,t)
   A = @view  u[:,:,1]
   B = @view  u[:,:,2]
   C = @view  u[:,:,3]
  dA = @view du[:,:,1]
  dB = @view du[:,:,2]
  dC = @view du[:,:,3]
  mul!(MyA,My,A)
  mul!(AMx,A,Mx)
  @. DA = DD*(MyA + AMx)
  @. dA = DA + aaa - β₁*A - r₁*A*B + r₂*C
  @. dB = α₂ - β₂*B - r₁*A*B + r₂*C
  @. dC = α₃ - β₃*C + r₁*A*B - r₂*C
end

u0 = zeros(N,N,3)

using BenchmarkTools
prob = ODEProblem(f,u0,(0.0,T))
#@btime sol = solve(prob, ROCK4(), reltol = 1e-8, abstol=1e-8, saveat=dt);
sol = solve(prob, ROCK4(), reltol = 1e-8, abstol=1e-8, saveat=dt); # 457.078 ms (3195 allocations: 238.33 MiB)

ode_data = Array(sol);

########################
# Define the neural PDE
########################
n_weights = 10

#for the reaction term
rx_nn = Chain(Dense(2, n_weights, tanh),
                Dense(n_weights, 2*n_weights, tanh),
                Dense(2*n_weights, n_weights, tanh),
                Dense(n_weights, 2))


#conv with bias with initial values as 1/dx^2
w_err = 0.0
init_w = reshape([1.1 -2.5 1.0], (3, 1, 1, 1))
diff_cnn_ = Conv(init_w, [0.], pad=(0,0,0,0))

#initialize D0 close to D/dx^2
D0 = [6.5]


p1,re1 = Flux.destructure(rx_nn)
p2,re2 = Flux.destructure(diff_cnn_)
p1
p2
p = [p1;p2;D0]
full_restructure(p) = re1(p[1:length(p1)]), re2(p[(length(p1)+1):end-1]), p[end]
p = p .*p'

function nn_ode(u,p,t)
    rx_nn = re1(p[1:length(p1)])

    u_cnn_1   = [p[end-4,end-4] * u[end,end,k] + p[end-3,end-3] * u[1,1,k] + p[end-2,end-2] * u[2,2,k] for k in 1:3]
    u_cnn     = [p[end-4,end-4] * u[i-1,j-1,k] + p[end-3,end-3] * u[i,j,k] + p[end-2,end-2] * u[i+1,j+1,k] for i in 2:Nx-1, j in 2:Ny-1, k in 1:3]
    u_cnn_end = [p[end-4,end-4] * u[end-1,end-1,k] + p[end-3,end-3] * u[end,end,k] + p[end-2,end-2] * u[1,1,k] for k in 1:3]

    [rx_nn([u[i,j,k], u[j,i,k]])[1] for i in 1:Nx, j in 1:Ny, k in 1:3] + p[end][end] * cat(reshape(u_cnn_1, (1, 1, 3)), u_cnn, reshape(u_cnn_end, (1, 1, 3)); dims = (1,2))
end


########################
# Solving the neural PDE and setting up loss function
########################

prob_nn = ODEProblem(nn_ode, u0, (0.0, T), p)

sol_nn = concrete_solve(prob_nn,VCABM(), u0, p, sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP()) )

function predict_rd(θ)
  # No ReverseDiff if using Flux
    Array(concrete_solve(prob_nn,VCABM(),u0,θ,saveat=dt,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
end

#match data and force the weights of the CNN to add up to zero
function loss_rd(p)
    pred = predict_rd(p)
    sum(abs2, ode_data .- pred) + 10^2 * abs(sum(p[end-4 : end-2, end-4:end-2])), pred
end


########################
# Training
########################

#Optimizer
opt = ADAM(0.05)

global count = 0
global save_count = 0
save_freq = 50

train_arr = Float64[]
diff_arr = Float64[]
w1_arr = Float64[]
w2_arr = Float64[]
w3_arr = Float64[]


res1 = DiffEqFlux.sciml_train(loss_rd, p, opt, maxiters = 1, progress = true)

res2 = DiffEqFlux.sciml_train(loss_rd, res1.minimizer, opt, cb=cb, maxiters = 300)
res3 = DiffEqFlux.sciml_train(loss_rd, res2.minimizer, BFGS(), cb=cb, maxiters = 1000)

pstar = res3.minimizer
