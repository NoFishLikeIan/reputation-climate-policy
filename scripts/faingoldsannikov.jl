## Setup
using Revise
using Optim
using UnPack
using FastClosures
using BenchmarkTools
using Printf
using JLD2

using DifferentialEquations, BoundaryValueDiffEq
using SciMLBase: successful_retcode

includet("../src/primitives/constants.jl")
includet("../src/primitives/signal.jl")

includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")

includet("../src/solve/equilibrium.jl")
includet("../src/solve/valuefunction.jl")

## Defaults
firm = Firm()
government = Government()
signal = Signal()

τᶜ = computeτᶜ(government, firm)

## Solve value function
parameters = (τᶜ, signal, government, firm)
φsteps = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4]
φstep = first(φsteps)
φspan = (φstep, 1 - φstep)
ℓspan = logit.(φspan)

u₀ = w(0., 0., government, firm)
u₁ = w(0., aᶜ(τᶜ, firm), government, firm)

∂u = u₁ - u₀
α = leftboundaryexponent(parameters)
@printf "Left boundary exponent α = %.4f\n" α

xinit = (parameters, ℓ) -> begin
    τᶜ, _, government, firm = parameters

    u₀ = w(0., 0., government, firm)
    u₁ = w(0., aᶜ(τᶜ, firm), government, firm)

    ∂u = u₁ - u₀
    α = leftboundaryexponent(parameters)
    φ = belief(ℓ)

    û = u₀ + ∂u * φ^α
    ẑ = -α * ∂u * φ^α * (1 - φ) / government.r

    return [û, ẑ]
end

function continuationguess(previoussol, previousstep, parameters)
    τᶜ, _, government, firm = parameters
    u₀ = w(0., 0., government, firm)
    u₁ = w(0., aᶜ(τᶜ, firm), government, firm)
    α = leftboundaryexponent(parameters)

    leftx = previoussol(logit(previousstep))
    rightx = previoussol(logit(1 - previousstep))
    leftz = leftx[2]
    rightz = rightx[2]

    return (_, ℓ) -> begin
        φ = belief(ℓ)

        if φ < previousstep
            ẑ = leftz * (φ / previousstep)^α
            û = u₀ - government.r * ẑ / α
            return [û, ẑ]
        elseif φ > 1 - previousstep
            ẑ = rightz * (1 - φ) / previousstep
            û = u₁ + government.r * ẑ
            return [û, ẑ]
        else
            return previoussol(ℓ)
        end
    end
end

## Test the ODE
x₀ = xinit(parameters, first(ℓspan))
dx₀ = similar(x₀)

Flogit!(dx₀, x₀, parameters, first(ℓspan))
odeprob = ODEProblem(Flogit!, xinit, ℓspan, parameters)

## Solve TwoPointBVProblem
bcresiduals = (zeros(1), zeros(1))
solutions = Tuple{Float64, Vector{Float64}, Vector{Vector{Float64}}}[]
global guess = xinit

for (i, φstep) in enumerate(φsteps)
    @printf "Solving problem %d/%d with φ = %.2e\n" i length(φsteps) φstep

    local φspan = (φstep, 1 - φstep)
    local ℓspan = logit.(φspan)
    local ℓstep = min(0.05, (last(ℓspan) - first(ℓspan)) / 200)
    prob = TwoPointBVProblem(Flogit!, (leftboundary!, rightboundary!), guess, ℓspan, parameters; bcresid_prototype = bcresiduals)
    sol = solve(prob, MIRK4(); dt = ℓstep, abstol = 1e-6, reltol = 1e-6)

    if !successful_retcode(sol.retcode)
        @warn "BVP failed at φstep = $(φstep) with retcode $(sol.retcode)"
    end

    push!(solutions, (φstep, sol.t, sol.u))
    global guess = continuationguess(sol, φstep, parameters)
end

JLD2.@save "data/solutions/continuous-time.jld2" solutions τᶜ signal government firm