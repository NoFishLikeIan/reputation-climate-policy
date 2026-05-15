## Setup
using Revise
using Optim
using UnPack
using BenchmarkTools

using DifferentialEquations, BoundaryValueDiffEq

includet("primitives/constants.jl")
includet("primitives/signal.jl")

includet("agents/firm.jl")
includet("agents/government.jl")

includet("solve/equilibrium.jl")
includet("solve/valuefunction.jl")

## Defaults
firm = Firm()
government = Government()
signal = Signal()

τᶜ = computeτᶜ(government, firm)

## Solve value function
parameters = (τᶜ, signal, government, firm)
φstep = 1e-4
φspan = (φstep, 1 - φstep)

u₀ = w(0., 0., government, firm)
u₁ = w(τᶜ, aᶜ(τᶜ, firm), government, firm)

∂u = (u₁ - u₀) / (1 - 2φstep)

function initialisex(parameters, φ)
    τᶜ, _, government, firm = parameters

    u₀ = w(0., 0., government, firm)
    u₁ = w(τᶜ, aᶜ(τᶜ, firm), government, firm)

    ∂u = (u₁ - u₀)

    û = u₁ + ∂u * φ
    ẑ = -∂u * φ * (1 - φ) / government.r

    return [û, ẑ]
end

## Test the ODE
x₀ = initialisex(parameters, φstep)
dx₀ = similar(x₀)

F!(dx₀, x₀, parameters, φstep)
odeprob = ODEProblem(F!, initialisex, φspan, parameters)

## Solve TwoPointBVProblem
prob = TwoPointBVProblem(F!, (leftboundary!, rightboundary!), initialisex, φspan, parameters; bcresid_prototype = (zeros(1), zeros(1)))