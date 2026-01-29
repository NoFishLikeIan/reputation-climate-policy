using Revise
using BenchmarkTools
using FastClosures
using Base.Threads
using UnPack
using DifferentialEquations

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

function vadj(u, v, φ, signal, government, firm)
    @unpack α, σ = signal
    @unpack ρ = government

    τᶜ = committedtax(government, firm)
    τ = optimaltax(φ, v / ρ, signal, government, firm)
    a = optimalabatement(φ, v / ρ, signal, government, firm)
    wᵒ = w(τ, a, government, firm)

    return (2ρ * σ^2) * (u - wᵒ) / (α * (τᶜ - τ))^2
end

function F!(dz, z, p, x)
    signal, government, firm = p
    u, v = z
    φ = sigmoid(x)
    
    dz[1] = v
    dz[2] = v + vadj(u, v, φ, signal, government, firm)

    return dz
end

meanwelfare = (committed + stackleberg) / 2
z₀ = [meanwelfare, 1.]
Z = 30.
xspan = (-Z, Z)
p = (signal, government, firm)
bcresid_prototype = (zeros(2), zeros(2))

bvp = TwoPointBVProblem(F!, (leftbc!, rightbc!), z₀, xspan, p; bcresid_prototype)
sol = solve(bvp, RadauIIa7(), dt = 2Z / 10_000.)