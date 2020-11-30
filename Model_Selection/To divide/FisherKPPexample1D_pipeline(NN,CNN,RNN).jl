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

# Short description of this file:This script simulates the Fisher-KPP equation
#and fits a neural PDE to the data with the growth (aka reaction) term replaced
# by a feed-forward neural network and the diffusion term with a CNN.
#The resulting equation is solved with a RNN.

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
tTrain  = collect(0:dt:T/2);
tValid  = collect(T/2:dt:T);
Nx = Int64(X/dx+1);
Nt = Int64(T/dt+1);

#initial conditions
Amp = 1.0;
Delta = 0.2
#IC-1
rho0 = Amp*(tanh.((x .- (0.5 - Delta/2))/(Delta/10)) - tanh.((x .- (0.5 + Delta/2))/(Delta/10)))/2
#IC-2
#rho0 = Amp*(1 .- tanh.((x .- 0.2)/(Delta/6)))/2.

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
reaction(u) = r * u .* (1 .- u)
lap = diagm(0 => -2.0 * ones(Nx), 1=> ones(Nx-1), -1 => ones(Nx-1)) ./ dx^2
#Periodic BC
lap[1,end] = 1.0/dx^2
lap[end,1] = 1.0/dx^2
#Neumann BC
#lap[1,2] = 2.0/dx^2
#lap[end,end-1] = 2.0/dx^2

function rc_ode(rho, p, t)
    #finite difference
    D * lap * rho + reaction.(rho)
end

prob = ODEProblem(rc_ode, rho0, (0.0, T), saveat=dt)
sol = DifferentialEquations.solve(prob, RK4()) #Tsit5()


ode_data = Array(sol);

figure(figsize=(8,3))

subplot(121)
pcolor(x,t,ode_data')
xlabel("x"); ylabel("t");
colorbar()

subplot(122)
for i in 1:2:Nt
    plot(x, ode_data[:,i], label="t=$(sol.t[i])")
end
xlabel("x"); ylabel(L"$\rho$")
legend(frameon=false, fontsize=7, bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
tight_layout()
savefig(@sprintf("%s/training_data.pdf", save_folder))
gcf()

########################
# Define the neural PDE
########################
n_weights = 10

#for the reaction term
rx_nn = Chain(Dense(1, n_weights, tanh),
                Dense(n_weights, 2*n_weights, tanh),
                Dense(2*n_weights, n_weights, tanh),
                Dense(n_weights, 1),
                x -> x[1])



#conv with bias with initial values as 1/dx^2
w_err = 0.0
init_w = reshape([1.1 -2.5 1.0], (3, 1, 1, 1))
diff_cnn_ = Conv(init_w, [0.], pad=(0,0,0,0))

#initialize D0 close to D/dx^2
D0 = [6.5]

p1,re1 = Flux.destructure(rx_nn)
p2,re2 = Flux.destructure(diff_cnn_)
p = [p1;p2;D0]
full_restructure(p) = re1(p[1:length(p1)]), re2(p[(length(p1)+1):end-1]), p[end]

function nn_ode(u,p,t)
    rx_nn = re1(p[1:length(p1)])

    u_cnn_1   = [p[end-4] * u[end] + p[end-3] * u[1] + p[end-2] * u[2]]
    u_cnn     = [p[end-4] * u[i-1] + p[end-3] * u[i] + p[end-2] * u[i+1] for i in 2:Nx-1]
    u_cnn_end = [p[end-4] * u[end-1] + p[end-3] * u[end] + p[end-2] * u[1]]

    # Equivalent using Flux, but slower!
    #CNN term with periodic BC
    #diff_cnn_ = Conv(reshape(p[(end-4):(end-2)],(3,1,1,1)), [0.0], pad=(0,0,0,0))
    #u_cnn = reshape(diff_cnn_(reshape(u, (Nx, 1, 1, 1))), (Nx-2,))
    #u_cnn_1 = reshape(diff_cnn_(reshape(vcat(u[end:end], u[1:1], u[2:2]), (3, 1, 1, 1))), (1,))
    #u_cnn_end = reshape(diff_cnn_(reshape(vcat(u[end-1:end-1], u[end:end], u[1:1]), (3, 1, 1, 1))), (1,))

    [rx_nn([u[i]])[1] for i in 1:Nx] + p[end] * vcat(u_cnn_1, u_cnn, u_cnn_end)
end

########################
# Soving the neural PDE and setting up loss function
########################
prob_nn = ODEProblem(nn_ode, rho0, (0.0, T), p)
sol_nn = concrete_solve(prob_nn,Tsit5(), rho0, p)

function predict_rd(θ)
  # No ReverseDiff if using Flux
  Array(concrete_solve(prob_nn,Tsit5(),rho0,θ,saveat=dt,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
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
res1 = DiffEqFlux.sciml_train(loss_rd, p, ADAM(0.01), cb=cb, maxiters = 10)
res2 = DiffEqFlux.sciml_train(loss_rd, res1.minimizer, ADAM(0.01), cb=cb, maxiters = 50)
res3 = DiffEqFlux.sciml_train(loss_rd, res2.minimizer, BFGS(), cb=cb, maxiters = 100)

pstar = res3.minimizer


### Recurrent Neural Network

master_rnn = Flux.RNN(26, 1, (x -> x))

function predict_rnn(x)
  out = master_rnn.(x)[end]
  Flux.reset!(master_rnn)
  out
end


function loss_rnn(p)
    #pred = predict_rd(p)
    pred_rnn = predict_rnn(p)
    sum(abs2, ode_data .- pred_rnn), pred_rnn
end


ps = Flux.params(master_rnn)

train_data = [nn_ode(predict_rd(p)[:,k], p, tTrain) for k in 1:length(tTrain)]
test_data = [nn_ode(predict_rd(p)[:,k], p, tTrain) for k in length(tTrain):length(tTrain)+ length(tValid) -1]


train_labels = [predict_rd(p)[:,k] for k in 1:length(tTrain)]
test_labels = [predict_rd(p)[:,k] for k in length(tTrain):length(tTrain) + length(tValid) - 1]

loss(x, y) = abs(sum((predict_rnn(x) .- y)))

n_epochs = 50

@epochs n_epochs Flux.train!(loss, ps, zip(train_data, train_labels), opt)

using BenchmarkTools
@btime pred_test = [predict_rnn(test_data[k]) for k in 1:length(test_data)]

train_lab = reshape(train_labels[1], : , 1)
for k in 2:6
    train_lab = cat(train_lab, reshape(train_labels[k], : , 1), dims= (2))

end

fig = figure(figsize=(8,2.5));
subplot(131)
pcolormesh(x,tTrain,train_lab')
xlabel(L"$x$"); ylabel(L"$t$"); title("Data")
colorbar()

pred_test_fig = reshape(pred_test[1], : , 1)
for k in 2:6
    pred_test_fig = cat(pred_test_fig, reshape(pred_test[k], : , 1), dims= (2))

end


subplot(132)
img = pcolormesh(x,tValid,pred_test_fig')
global img
xlabel(L"$x$"); ylabel(L"$t$"); title("Prediction")
colorbar(); clim([0, 1]);

display(gcf())

## Save trained model
"""
function pipeline()
    u = u0
    for i in t
        uderiv = nn_ode(u, p, t)
        u = predict_rnn(zip(u, uderiv))
    end
end
"""
