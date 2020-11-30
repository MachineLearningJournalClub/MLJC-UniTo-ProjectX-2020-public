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

# Short description of this file:Burger's Equation

using DelimitedFiles,Plots
using DiffEqSensitivity, OrdinaryDiffEq, Zygote, Flux, DiffEqFlux, Optim

# Problem setup parameters:
Lx = 10.0
Ly = 10.0

x  = 0.0:0.01:Lx
y  = 0.0:0.01:Ly

x = collect(x)
y = collect(y)

dx = x[2] - x[1]
dy = y[2] - y[1]

Nx = size(x)
Ny = size(y)

u0 = zeros(Nx)
u0 = u0 * zeros(Ny)'


for i in 1:Nx[1]
    for j in 1:Ny[1]
        xv = (x[i]-5)
        yv = (y[j]-5)
        u0[i,j] = exp(-(((xv*xv+yv*yv)^(1/2)))^2)
    end
end

u0
v0 = deepcopy(u0) ./ 2


Plots.heatmap(u0)

display(gcf())
PyPlot.clf()



## Problem Parameters

#xtrs     = [dx,Nx]      # Extra parameters
# a0x, a0y, a1x, a1y, b0x, b0y, b1x, b1y = p
p        = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
dt       = 400.0*dx^2    # CFL condition
t0, tMax = 0.0 ,100*dt
tspan    = (t0,tMax)
t        = 0:dt:tMax
Re       = 2500.

## Definition of Auxiliary functions
# Central differences

function ddx(u,dx)
    """
    2nd order Central difference for 1st degree derivative
    """
    paddingLine = zeros(1,1001)
    matrix = (u[3:end, 1:end] - u[1:end-2, 1:end]) ./ (2.0*dx)
    matrix = cat(paddingLine, matrix; dims = (1))
    matrix = cat(matrix, paddingLine; dims = (1))
    return matrix
end

function d2dx(u,dx)
    """
    2nd order Central difference for 2nd degree derivative
    """
    paddingLine = zeros(1,1001)
    matrix = (u[3:end, 1:end] - 2.0.*u[2:end-1, 1:end] + u[1:end-2, 1:end]) ./ (dx^2)
    matrix = cat(paddingLine, matrix; dims = (1))
    matrix = cat(matrix, paddingLine; dims = (1))
    return matrix
end


function ddy(u,dy)
    """
    2nd order Central difference for 1st degree derivative
    """
    paddingLine = zeros(1001,1)
    matrix = (u[1:end, 3:end] - u[1:end, 1:end-2]) ./ (2.0*dy)
    matrix = cat(paddingLine, matrix; dims = (2))
    matrix = cat(matrix, paddingLine; dims = (2))
    return matrix
end



function d2dy(u,dy)
    """
    2nd order Central difference for 2nd degree derivative
    """
    paddingLine = zeros(1001,1)
    matrix = (u[1:end, 3:end] - 2.0.*u[1:end, 2:end-1] + u[1:end, 1:end-2]) ./ (dy^2)
    matrix = cat(paddingLine, matrix; dims = (2))
    matrix = cat(matrix, paddingLine; dims = (2))
    return matrix
end


## ODE description of the Physics:
function burgers(du,U,p,t)
    # Model parameters
    a0x, a0y, a1x, a1y, b0x, b0y, b1x, b1y = p
    #dx, Nx, dy, Ny
    u = U[1:1001, 1:1001]
    v = U[1002:end, 1002:end]
    #u = u[2]

    return [2.0*a0x .* u .* ddx(u, dx) +  2.0*a0y .* v .* ddy(u, dy)  -  a1x .* d2dx(u, dx) / Re -  a1y .* d2dy(u, dy) / Re ;
            2.0*b0x .* u .* ddx(v, dx) +  2.0*b0y .* v .* ddy(v, dy)  -  b1x .* d2dx(v, dx) / Re -  b1y .* d2dy(v, dy) / Re]
end


# Testing Solver on linear PDE
#U0 = [u0,v0]
U0 = cat(u0, v0 ; dims = (1,2))


prob = ODEProblem(burgers,U0,tspan,p)
<<<<<<< Updated upstream

sol = solve(prob,Tsit5(), dt=dt,saveat=t)
=======
prob
sol  = solve(prob,Tsit5(), dt=dt,saveat=t)
>>>>>>> Stashed changes

sol_burg = Array(sol)
u_sol = sol_burg[1:1001, 1:1001,1:end]
v_sol = sol_burg[1002:end, 1002:end, 1:end]

Plots.heatmap(u_sol[1:end,1:end,100])

display(gcf())
PyPlot.clf()


Plots.plot(x, sol.u[1], lw=3, label="t0", size=(800,500))
Plots.plot!(x, sol.u[end],lw=3, ls=:dash, label="tMax")

ps  = [0.1, 0.2];   # Initial guess for model parameters
function predict(θ)
    Array(solve(prob,Tsit5(),p=θ,dt=dt,saveat=t))
end

## Defining Loss function
function loss(θ)
    pred = predict(θ)
    l = predict(θ)  - sol
    return sum(abs2, l), pred # Mean squared error
end

l,pred   = loss(ps)
size(pred), size(sol), size(t) # Checking sizes

LOSS  = []                              # Loss accumulator
PRED  = []                              # prediction accumulator
PARS  = []                              # parameters accumulator

cb = function (θ,l,pred) #callback function to observe training
  display(l)
  append!(PRED, [pred])
  append!(LOSS, l)
  append!(PARS, [θ])
  false
end

cb(ps,loss(ps)...) # Testing callback function

# Let see prediction vs. Truth
Plots.scatter(sol[:,end], label="Truth", size=(800,500))
plot!(PRED[end][:,end], lw=2, label="Prediction")

res = DiffEqFlux.sciml_train(loss, ps, ADAM(0.01), cb = cb, maxiters = 100)  # Let check gradient propagation
ps = res.minimizer
res = DiffEqFlux.sciml_train(loss, ps, BFGS(), cb = cb, maxiters = 10,
                             allow_f_increases = false)  # Let check gradient propagation
@show res.minimizer # returns [0.999999999999975, 1.0000000000000213]
