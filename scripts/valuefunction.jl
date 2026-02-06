using Revise
using BenchmarkTools
using FastClosures
using Base.Threads
using UnPack
using DifferentialEquations, SparseArrays

using Plots, LaTeXStrings, Printf

Plots.default(linewidth = 2, dpi = 180, label = false, background_color = :transparent)
plotpath = "figures/preliminaries"; if !ispath(plotpath) mkpath(plotpath) end

includet("../src/constants.jl")
includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")
includet("../src/signal.jl")
includet("../src/optimal.jl")
includet("../src/utils.jl")

begin
	firm = Firm()
	government = Government()
    signal = Signal()

	τᶜ = committedtax(government, firm)
	
    stackleberg = w(0., 0., government, firm)
	committed = w(τᶜ, aᶜ(τᶜ, firm), government, firm)
end;

function leftbc!(res, z₀, p)
    _, government, firm = p
    u₀, v₀ = z₀
	
    res[1] = w(0., 0., government, firm) - u₀
	res[2] = v₀
end;

function rightbc!(res, z₁, p)
    _, government, firm = p
    u₁, v₁ = z₁
    
    res[1] = u₁ - w(τᶜ, aᶜ(τᶜ, firm), government, firm)
	res[2] = v₁
end;

function forcing(u, z, φ, signal, government, firm)
    @unpack α, σ = signal

    τᶜ = committedtax(government, firm)
    τ = optimaltax(φ, z, signal, government, firm)
    a = optimalabatement(φ, z, signal, government, firm)
    wᵒ = w(τ, a, government, firm)

    signal = (σ / (α * (τᶜ - τ)))^2

    return 2 * signal * (u - wᵒ)
end

function F!(dx, x, p, ψ)
    signal, government, firm = p
    u, z = x
    φ = sigmoid(ψ)
    
    dx[1] = government.r * z
    dx[2] = z + forcing(u, z, φ, signal, government, firm)

    return dx
end

p = (signal, government, firm)
meanwelfare = (committed + stackleberg) / 2
x₀ = [meanwelfare, 0.01]

let # Test function
    dx₀ = similar(x₀)
    ψ₀ = 0.

    @btime F!($dx₀, $x₀, $p, $ψ₀)
end;

Z = 10.
ψspan = (-Z, Z)
bcresid_prototype = (zeros(2), zeros(2))

bvp = TwoPointBVProblem(F!, (leftbc!, rightbc!), x₀, ψspan, p; bcresid_prototype);
sol = solve(bvp,  MIRK6(), dt = 0.001)