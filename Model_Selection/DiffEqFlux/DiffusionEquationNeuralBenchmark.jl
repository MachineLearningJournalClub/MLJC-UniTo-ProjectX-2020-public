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
#redefinitions of constants.

cd(@__DIR__)
using Pkg#; Pkg.activate("."); Pkg.instantiate()
Pkg.add("BSON")
Pkg.add("DifferentialEquations")


using PyPlot, Printf
using LinearAlgebra
using Flux, DiffEqFlux, Optim, DiffEqSensitivity
using BSON: @save, @load
using Flux: @epochs
using DifferentialEquations

#parameter
D = 0.01; #diffusion
r = 1.0; #reaction rate

#domain
X = 1.0; T = 5.0;
dx = 0.04; dt = T/10;
x = collect(0:dx:X);
t = collect(0:dt:T);
Nx = 128 # Int64(X/dx+1);
Ny = 128 #Int64(Y/dy+1);
Nt = Int64(T/dt+1);

save_folder = "data"

if isdir(save_folder)
    rm(save_folder, recursive=true)
end
mkdir(save_folder)

close("all")
figure()
plot(x, rho0)
title("Initial Condition")
gcf()

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
u0 = u0 .- u0
prob = ODEProblem(f,u0,(0.0,10.0))



prob = ODEProblem(f,u0,(0.0,10.0))
sol = solve(prob, ROCK4(), reltol = 1e-8, abstol=1e-8, saveat=0.1);


ode_data = Array(sol);

########################
# Define the neural PDE
########################
n_weights = 10

#for the reaction term
rx_nn = Chain(Dense(2, n_weights, tanh),
                Dense(n_weights, 2*n_weights, tanh),
                Dense(2*n_weights, n_weights, tanh),
                Dense(n_weights, 2))#,
            #    x -> x)

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
p = p .- p'
re1 #Ha usato una regex

function nn_ode(u,p,t)
    rx_nn = re1(p[1:length(p1)])

    u_cnn_1   = [p[end-4,end-4] * u[end,end,k] + p[end-3,end-3] * u[1,1,k] + p[end-2,end-2] * u[2,2,k] for k in 1:3]
    u_cnn     = [p[end-4,end-4] * u[i-1,j-1,k] + p[end-3,end-3] * u[i,j,k] + p[end-2,end-2] * u[i+1,j+1,k] for i in 2:Nx-1, j in 2:Ny-1, k in 1:3]
    u_cnn_end = [p[end-4,end-4] * u[end-1,end-1,k] + p[end-3,end-3] * u[end,end,k] + p[end-2,end-2] * u[1,1,k] for k in 1:3]

    print(size(u_cnn_1))
    print(size(u_cnn_end))

    # Equivalent using Flux, but slower!
    #CNN term with periodic BC
    #diff_cnn_ = Conv(reshape(p[(end-4):(end-2),(end-4):(end-2)],(3, 3, 1,1)), [0.0], pad=(0,0,0,0))
    #temp2 = reshape(u, (Nx, Ny,1,3))
    #temp = diff_cnn_(temp2)
    #print(size(temp))
    #u_cnn = reshape(temp, (Nx-2,Ny-2,1,3))
    #u_cnn_1 = reshape(diff_cnn_(reshape(vcat(u[end:end], u[1:1], u[2:2]), (3, 1, 1, 1))), (1,))
    #u_cnn_end = reshape(diff_cnn_(reshape(vcat(u[end-1:end-1], u[end:end], u[1:1]), (3, 1, 1, 1))), (1,))
    # print(u)
    print(size(u_cnn_1))
    [rx_nn([u[i,j,k]]) for i in 1:Nx-2, j in 1:Ny-2, k in 1:3] + p[end][end] * vcat(ones(1,126) .- u_cnn_1', u_cnn, ones(1,126) .-  u_cnn_end')
end

########################
# Soving the neural PDE and setting up loss function
########################
prob_nn = ODEProblem(nn_ode, u0, (0.0, T), p)
sol_nn = concrete_solve(prob,Tsit5(), u0, p)

function predict_rd(θ)
  # No ReverseDiff if using Flux
    Array(concrete_solve(prob_nn,Tsit5(),u0,θ,saveat=dt,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
end

#match data and force the weights of the CNN to add up to zero
function loss_rd(p)
    pred = predict_rd(p)
    sum(abs2, ode_data .- pred) + 10^2 * abs(sum(p[end-4 : end-2])), pred
end

########################
# Training
########################

#Optimizer
opt = ADAM(0.001)

global count = 0
global save_count = 0
save_freq = 50

train_arr = Float64[]
diff_arr = Float64[]
w1_arr = Float64[]
w2_arr = Float64[]
w3_arr = Float64[]

#callback function to observe training
cb = function (p,l,pred)
    rx_nn, diff_cnn_, D0 = full_restructure(p)
    push!(train_arr, l)
    push!(diff_arr, p[end])

    weight = diff_cnn_.weight[:]
    push!(w1_arr, weight[1])
    push!(w2_arr, weight[2])
    push!(w3_arr, weight[3])

    println(@sprintf("Loss: %0.4f\tD0: %0.4f Weights:(%0.4f,\t %0.4f, \t%0.4f) \t Sum: %0.4f"
            ,l, D0[1], weight[1], weight[2], weight[3], sum(weight)))

    global count

    if count==0
        fig = figure(figsize=(8,2.5));
        ttl = fig.suptitle(@sprintf("Epoch = %d", count), y=1.05)
        global ttl
        subplot(131)
        pcolormesh(x,t,ode_data')
        xlabel(L"$x$"); ylabel(L"$t$"); title("Data")
        colorbar()

        subplot(132)
        img = pcolormesh(x,t,pred')
        global img
        xlabel(L"$x$"); ylabel(L"$t$"); title("Prediction")
        colorbar(); clim([0, 1]);

        ax = subplot(133); global ax
        u = collect(0:0.01:1)
        rx_line = plot(u, rx_nn.([[elem] for elem in u]), label="NN")[1];
        global rx_line
        plot(u, reaction.(u), label="True")
        title("Reaction Term")
        legend(loc="upper right", frameon=false, fontsize=8);
        ylim([0, r*0.25+0.2])

        subplots_adjust(top=0.8)
        tight_layout()
    end

    if count>0
        println("updating figure")
        img.set_array(pred[1:end-1, 1:end-1][:])
        ttl.set_text(@sprintf("Epoch = %d", count))

        u = collect(0:0.01:1)
        rx_pred = rx_nn.([[elem] for elem in u])
        rx_line.set_ydata(rx_pred)
        u = collect(0:0.01:1)

        min_lim = min(minimum(rx_pred), minimum(reaction.(u)))-0.1
        max_lim = max(maximum(rx_pred), maximum(reaction.(u)))+0.1

        ax.set_ylim([min_lim, max_lim])
    end

    global save_count
    if count%save_freq == 0
        println("saved figure")
        savefig(@sprintf("%s/pred_%05d.png", save_folder, save_count), dpi=200, bbox_inches="tight")
        save_count += 1
    end

    display(gcf())
    count += 1

    false
end

#train
res1 = DiffEqFlux.sciml_train(loss_rd, p, ADAM(0.001), cb=cb, maxiters = 100)
res2 = DiffEqFlux.sciml_train(loss_rd, res1.minimizer, ADAM(0.001), cb=cb, maxiters = 300)
res3 = DiffEqFlux.sciml_train(loss_rd, res2.minimizer, BFGS(), cb=cb, maxiters = 1000)

pstar = res3.minimizer

## Save trained model
@save @sprintf("%s/model.bson", save_folder) pstar

########################
# Plot for PNAS paper
########################
@load @sprintf("%s/model.bson", save_folder) pstar
#re-defintions for newly loaded data

diff_cnn_ = Conv(reshape(pstar[(end-4):(end-2)],(3,1,1,1)), [0.0], pad=(0,0,0,0))
diff_cnn(x) = diff_cnn_(x) .- diff_cnn_.bias
D0 = res3.minimizer[end]

fig = figure(figsize=(4,4))

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 10
rcParams["text.usetex"] = true
rcParams["font.family"] = "serif"
rcParams["font.sans-serif"] = "Helvetica"
rcParams["axes.titlesize"] = 10

subplot(221)
pcolormesh(x,t,ode_data', rasterized=true)
xlabel(L"$x$"); ylabel(L"$t$"); title("Data")
yticks([0, 1, 2, 3, 4, 5])

ax = subplot(222)
cur_pred = predict_rd(pstar)[1]
img = pcolormesh(x,t,cur_pred', rasterized=true)
global img
xlabel(L"$x$"); ylabel(L"$t$"); title("Prediction")
yticks([0, 1, 2, 3, 4, 5])
cax = fig.add_axes([.48,.62,.02,.29])
colb = fig.colorbar(img, cax=cax)
colb.ax.set_title(L"$\rho$")
clim([0, 1]);
colb.set_ticks([0, 1])

subplot(223)
plot(Flux.data(w1_arr ./ w3_arr) .- 1, label=L"$w_1/w_3 - 1$")
plot(Flux.data(w1_arr .+ w2_arr .+ w3_arr), label=L"$w_1 + w_2 + w_3$")
axhline(0.0, linestyle="--", color="k")
xlabel("Epochs"); title("CNN Weights")
xticks([0, 1500, 3000]); yticks([-0.4, -0.3,-0.2, -0.1, 0.0, 0.1])
legend(loc="lower right", frameon=false, fontsize=6)

subplot(224)
u = collect(0:0.01:1)
plot(u, rx_nn.([[elem] for elem in u]), label="UPDE")[1];
plot(u, reaction.(u), linestyle="--", label="True")
xlabel(L"$\rho$")
title("Reaction Term")
legend(loc="lower center", frameon=false, fontsize=6);
ylim([0, 0.3])

tight_layout(h_pad=1)
gcf()
savefig(@sprintf("%s/fisher_kpp.pdf", save_folder))

#plot loss vs epochs and save
figure(figsize=(6,3))
plot(log.(train_arr), "k.", markersize=1)
xlabel("Epochs"); ylabel("Log(loss)")
tight_layout()
savefig(@sprintf("%s/loss_vs_epoch.pdf", save_folder))
gcf()
