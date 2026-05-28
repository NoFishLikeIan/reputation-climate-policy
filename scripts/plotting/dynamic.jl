using Revise
using BenchmarkTools
using Printf

import UnPack: @unpack
import FastClosures: @closure

import Optim
import JLD2

import Roots
import FastInterpolations

import SciMLBase
import OrdinaryDiffEqSDIRK
import BoundaryValueDiffEq as BVP
import OrdinaryDiffEq as ODE
import StochasticDiffEq as SDE
import RecursiveArrayTools as RA


using LaTeXStrings, Printf
using Colors

import Statistics
import Plots

Plots.default(linewidth = 2.5, dpi = 180, size = 550 .* (√2, 1))
includet("colors.jl")

includet("../../src/utils/saving.jl")
includet("../../src/primitives/constants.jl")
includet("../../src/primitives/signal.jl")

includet("../../src/agents/firm.jl")
includet("../../src/agents/government.jl")

includet("../../src/solve/equilibrium.jl")
includet("../../src/solve/valuefunction.jl")
includet("../../src/solve/staticproblem.jl")
includet("../../src/solve/dynamicvaluefunction.jl")

firm = DynamicFirm(ω = 5e-2, ν = 4e-2)
solutionpath = joinpath("data", "solutions", dynamicsolutionlabel(firm))

if !isfile(solutionpath)
    error("No dynamic solution found at $(solutionpath). Run scripts/dynamic.jl with the same firm parameters first.")
end

JLD2.@load solutionpath solution ℓgrid tgrid τᶜtraj signal government firm

plotpath = joinpath("figures", "dynamic", dynamicsolutionlabel(firm))
if !ispath(plotpath) mkpath(plotpath) end

## Solution GIF
φgrid = belief.(ℓgrid)

anim = Plots.@animate for (i, t) in enumerate(tgrid)
    print("Plotting $(round(t, digits = 0)) / $(tgrid[end])\r")
    τᶜᵢ = τᶜtraj[i]
    uᵢ = solution(t)

    Plots.Plots.plot(φgrid, uᵢ; ylims = (0., 8.), c = :black, xlabel = L"\varphi", xlims = (0, 1), ylabel = L"Value $u_t$", label = false, title = "Value at time t = $(round(t; digits = 2))")
end

Plots.gif(anim, joinpath(plotpath, "solution.gif"), fps = 30)

## Define dynamic policies
const clampextrap = FastInterpolations.ClampExtrap()
dynamicparameters = (solution, ℓgrid, tgrid, τᶜtraj, signal, government, firm);

"Computes the linear interpolation of the finite difference first and second derivative of `u` at time `t` and belief `φ`."
function valuedifferential(t, φ, dynamicparameters)
    solution, ℓgrid = dynamicparameters[1:2]
    ℓ = logit(φ)
    uₜ = solution(t)
    n = length(ℓgrid)

    if ℓ ≤ ℓgrid[2]
        return gridderivatives(uₜ, ℓgrid, 2)
    elseif ℓ ≥ ℓgrid[end - 1]
        return gridderivatives(uₜ, ℓgrid, n - 1)
    end

    rightindex = searchsortedfirst(ℓgrid, ℓ)
    leftindex = rightindex - 1

    uₗleft, uₗₗleft = gridderivatives(uₜ, ℓgrid, leftindex)
    uₗright, uₗₗright = gridderivatives(uₜ, ℓgrid, rightindex)
    
    weight = (ℓ - ℓgrid[leftindex]) / (ℓgrid[rightindex] - ℓgrid[leftindex])
    uₗ = (1 - weight) * uₗleft + weight * uₗright
    uₗₗ = (1 - weight) * uₗₗleft + weight * uₗₗright

    return uₗ, uₗₗ
end

"Computes the optimal tax `τ` t time `t` and belief `φ`."
function dynamictax(t, φ, dynamicparameters)
    _, _, tgrid, τᶜtraj, signal, government, firm = dynamicparameters
    τᶜ = FastInterpolations.linear_interp(tgrid, τᶜtraj, t; extrap = clampextrap)
    uₗ, uₗₗ = valuedifferential(t, φ, dynamicparameters)

    return ηᵈ(t, φ, uₗ, uₗₗ, τᶜ, signal, government, firm) * τᶜ
end

"Computes the optimal abatement `a` at time `t` and belief `φ`."
function dynamicabatement(t, φ, dynamicparameters)
    _, _, tgrid, τᶜtraj, _, _, firm = dynamicparameters
    τᶜ = FastInterpolations.linear_interp(tgrid, τᶜtraj, t; extrap = clampextrap)
    τ = dynamictax(t, φ, dynamicparameters)

    return aᵇ(t, τ, φ, τᶜ, firm)
end

begin # Plot optimal tax
    tsamples = (0., 10., 100.)
    timecolors = Plots.palette(:viridis, length(tsamples))
    taxfigure = Plots.Plots.plot(xlims = (0, 1), xlabel = L"\phi", ylabel = L"\tau", legendtitle = L"t")


    for (i, t) in enumerate(tsamples)
        τᶜ = FastInterpolations.linear_interp(tgrid, τᶜtraj, t; extrap = clampextrap)
        Plots.hline!([τᶜ]; c = timecolors[i], linestyle = :dash, label = false)
        Plots.plot!(taxfigure, φgrid, φ -> dynamictax(t, φ, dynamicparameters); label = t, c = timecolors[i])
    end

    taxfigure
end

begin # Plot optimal abatement
    abatementfigure = Plots.plot(xlims = (0, 1), xlabel = L"\phi", ylabel = L"\tau", legendtitle = L"t")

    for (i, t) in enumerate(tsamples)
        τᶜ = FastInterpolations.linear_interp(tgrid, τᶜtraj, t; extrap = clampextrap)
        
        Plots.hline!([aᶜ(t, τᶜ, firm)]; c = timecolors[i], linestyle = :dash, label = false)
        Plots.plot!(abatementfigure, φgrid, φ -> dynamicabatement(t, φ, dynamicparameters); label = t, c = timecolors[i])
    end

    abatementfigure
end


## Define belief dynamics function
function precision(t, φ, dynamicparameters)
    _, _, tgrid, τᶜtraj, signal, _, _ = dynamicparameters
    τᶜ = FastInterpolations.linear_interp(tgrid, τᶜtraj, t; extrap = clampextrap)
    τ = dynamictax(t, φ, dynamicparameters)

    return signal.ϵ * (τᶜ - τ) / signal.σ
end

function driftbeliefs(φ, dynamicparameters, t)
    ξ = precision(t, φ, dynamicparameters)

    return -φ^2 * (1 - φ) * ξ^2
end

function variancebeliefs(φ, dynamicparameters, t)
    ξ = precision(t, φ, dynamicparameters)

    return φ * (1 - φ) * ξ
end

begin # Plot drift E[dϕ]
    driftfigure = Plots.plot(xlims = (0, 1), xlabel = L"\phi", ylabel = L"\dot{\phi}", legendtitle = L"t")

    for t in (0., 1., 5.)
        Plots.plot!(driftfigure, φgrid, φ -> driftbeliefs(φ, dynamicparameters, t); label = t)
    end

    driftfigure
end

begin # Plot variance V[dϕ]
    variancefigure = Plots.plot(xlims = (0, 1), xlabel = L"\phi", ylabel = L"\dot{\phi}", legendtitle = L"t")

    for t in (0., 1., 5.)
        Plots.plot!(variancefigure, φgrid, φ -> variancebeliefs(φ, dynamicparameters, t); label = t)
    end

    variancefigure
end

## Compute belief trajectory
monthlytime = 0:(1 / 12):80
beliefinitialconditions = [0.2, 0.4, 0.6, 0.8]
isoutofdomain = @closure (φ, p, t) -> !(0 < φ < 1)

beliefproblem = SDE.SDEProblem(driftbeliefs, variancebeliefs, 1 / 2, extrema(monthlytime), dynamicparameters; isoutofdomain = isoutofdomain)
SDE.solve(beliefproblem, SDE.SRIW2()) # Checks that the model runs

beliefensembleproblem = SDE.EnsembleProblem(beliefproblem)

function timepointtaxabatementquantiles(sim, quantilelevels, t, dynamicparameters)
    _, _, tgrid, τᶜtraj, _, _, firm = dynamicparameters
    τᶜ = FastInterpolations.linear_interp(tgrid, τᶜtraj, t; extrap = clampextrap)

    taxvalues = Float64[]
    abatementvalues = Float64[]
    sizehint!(taxvalues, length(sim.u))
    sizehint!(abatementvalues, length(sim.u))

    for trajectory in sim.u
        φ = trajectory(t)
        τ = dynamictax(t, φ, dynamicparameters)
        a = aᵇ(t, τ, φ, τᶜ, firm)

        push!(taxvalues, τ)
        push!(abatementvalues, a)
    end

    return Statistics.quantile(taxvalues, quantilelevels), Statistics.quantile(abatementvalues, quantilelevels)
end

function timeseriestaxabatementquantiles(sim, quantilelevels, times, dynamicparameters)
    taxseries = Vector{Vector{Float64}}(undef, length(times))
    abatementseries = similar(taxseries)

    for (i, t) in enumerate(times)
        taxseries[i], abatementseries[i] = timepointtaxabatementquantiles(sim, quantilelevels, t, dynamicparameters)
    end

    return RA.DiffEqArray(taxseries, times), RA.DiffEqArray(abatementseries, times)
end

trajectories = 1000
quantilelevels = [0.05, 0.5, 0.95]
quantiles = RA.DiffEqArray[]
taxquantiles = RA.DiffEqArray[]
abatementquantiles = RA.DiffEqArray[]

for φ₀ in beliefinitialconditions
    sim = SDE.solve(beliefensembleproblem; u0 = φ₀, trajectories)

    summ = SDE.EnsembleAnalysis.timeseries_point_quantile(sim, quantilelevels, monthlytime)
    taxsumm, abatementsumm = timeseriestaxabatementquantiles(sim, quantilelevels, monthlytime, dynamicparameters)

    push!(quantiles, summ)
    push!(taxquantiles, taxsumm)
    push!(abatementquantiles, abatementsumm)
end

begin
    belieftrajectoryfig = Plots.plot(
        xlabel = L"Year $t$",
        ylabel = L"Belief $\phi_t$",
        ylims = (0, 1),
        legendtitle = L"\phi_0",
    )

    trajectorycolors = Plots.palette(:viridis, length(beliefinitialconditions))

    for (i, φ₀) in enumerate(beliefinitialconditions)
        summ = quantiles[i]

        Plots.plot!(belieftrajectoryfig, summ.t, getindex.(summ.u, 2); c = trajectorycolors[i], label = @sprintf("%.1f", φ₀))

        Plots.plot!(belieftrajectoryfig, summ.t, getindex.(summ.u, 1);
            fillrange = getindex.(summ.u, 3),
            linewidth = 0,
            color = trajectorycolors[i],
            fillalpha = 0.16, label = false
        )
    end

    Plots.savefig(belieftrajectoryfig, joinpath(plotpath, "belief-trajectories.png"))

    belieftrajectoryfig
end

## Tax and abatement trajectories
policyfigures = Plots.Plot[]

begin
    taxtrajectoryfig = Plots.plot(
        xlabel = L"Year $t$",
        ylabel = L"Tax $\tau_t$",
        legendtitle = L"\phi_0",
    )

    for (i, φ₀) in enumerate(beliefinitialconditions)
        summ = taxquantiles[i]

        Plots.plot!(taxtrajectoryfig, summ.t, getindex.(summ.u, 2); c = trajectorycolors[i], label = @sprintf("%.1f", φ₀))

        Plots.plot!(taxtrajectoryfig, summ.t, getindex.(summ.u, 1);
            fillrange = getindex.(summ.u, 3),
            linewidth = 0,
            color = trajectorycolors[i],
            fillalpha = 0.16, label = false
        )
    end

    Plots.savefig(taxtrajectoryfig, joinpath(plotpath, "tax-trajectories.png"))

    push!(policyfigures, taxtrajectoryfig)

    taxtrajectoryfig
end

begin
    abatementtrajectoryfig = Plots.hline([firm.e₀];
        xlabel = L"Year $t$",
        ylabel = L"Abatement $a_t$",
        legendtitle = L"\phi_0",
        label = false, c = :black, linestyle = :dash
    )

    for (i, φ₀) in enumerate(beliefinitialconditions)
        summ = abatementquantiles[i]

        Plots.plot!(abatementtrajectoryfig, summ.t, getindex.(summ.u, 2); c = trajectorycolors[i], label = @sprintf("%.1f", φ₀))

        Plots.plot!(abatementtrajectoryfig, summ.t, getindex.(summ.u, 1);
            fillrange = getindex.(summ.u, 3),
            linewidth = 0,
            color = trajectorycolors[i],
            fillalpha = 0.16, label = false
        )
    end

    Plots.savefig(abatementtrajectoryfig, joinpath(plotpath, "abatement-trajectories.png"))

    push!(policyfigures, abatementtrajectoryfig)
    
    abatementtrajectoryfig
end


dynamicpolicyfig = Plots.plot(
    policyfigures...;
    layout = (1, 2),
    size = 500 .* (2√2, 1),
    margins = 6Plots.mm,
)

Plots.savefig(dynamicpolicyfig, joinpath(plotpath, "tax-abatement-policy-grid.png"))

dynamicpolicyfig
