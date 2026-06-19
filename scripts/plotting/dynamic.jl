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

firm = DynamicFirm(ν = ν₀ * 0.3)
solutionpath = joinpath("data", "solutions", dynamicsolutionlabel(firm))

if !isfile(solutionpath)
    error("No dynamic solution found at $(solutionpath). Run scripts/dynamic.jl with the same firm parameters first.")
end

solutiondata = JLD2.load(solutionpath)
solutiongrid = solutiondata["solutiongrid"]
sgrid = solutiondata["sgrid"]
φgrid = solutiondata["φgrid"]
tgrid = solutiondata["tgrid"]
τᶜtraj = solutiondata["τᶜtraj"]
signal = solutiondata["signal"]
government = solutiondata["government"]
firm = solutiondata["firm"]
terminaltime = last(sgrid)

plotpath = joinpath("figures", "dynamic", dynamicsolutionlabel(firm))
if !ispath(plotpath) mkpath(plotpath) end

## Solution GIF
ufig = Plots.plot(xlabel = L"\varphi", xlims = (0, 1), ylabel = L"Value $u_t$", legendtitle = L"t")
for t in (0., 50., 100.)
    s = terminaltime - t
    rightindex = searchsortedfirst(sgrid, s)
    if rightindex <= firstindex(sgrid)
        uᵢ = solutiongrid[firstindex(sgrid), :]
    elseif rightindex > lastindex(sgrid)
        uᵢ = solutiongrid[lastindex(sgrid), :]
    else
        leftindex = rightindex - 1
        weight = (s - sgrid[leftindex]) / (sgrid[rightindex] - sgrid[leftindex])
        uᵢ = (1 - weight) * solutiongrid[leftindex, :] + weight * solutiongrid[rightindex, :]
    end

    Plots.plot!(ufig, φgrid, uᵢ; label = round(t; digits = 2))
end

ufig

## Define dynamic policies
const clampextrap = FastInterpolations.ClampExtrap()
dynamicparameters = (solutiongrid, sgrid, φgrid, tgrid, τᶜtraj, signal, government, firm);

function dynamicvalue(t, dynamicparameters)
    solutiongrid, sgrid = dynamicparameters[1:2]
    s = last(sgrid) - t
    rightindex = searchsortedfirst(sgrid, s)

    if rightindex <= firstindex(sgrid)
        return solutiongrid[firstindex(sgrid), :]
    elseif rightindex > lastindex(sgrid)
        return solutiongrid[lastindex(sgrid), :]
    end

    leftindex = rightindex - 1
    weight = (s - sgrid[leftindex]) / (sgrid[rightindex] - sgrid[leftindex])

    return (1 - weight) * solutiongrid[leftindex, :] + weight * solutiongrid[rightindex, :]
end

"Computes the linear interpolation of the generator curvature term at time `t` and belief `φ`."
function valuedifferential(t, φ, dynamicparameters)
    _, _, φgrid = dynamicparameters[1:3]
    n = length(φgrid)
    uₜ = dynamicvalue(t, dynamicparameters)
    uφgrid = [derivative(uₜ, φgrid, i) for i in eachindex(φgrid)]
    uφφgrid = [derivative(uφgrid, φgrid, i) for i in eachindex(φgrid)]
    Dgrid = [
        2 * φgrid[i] * (1 - φgrid[i]) * (-φgrid[i] * uφgrid[i] + φgrid[i] * (1 - φgrid[i]) * uφφgrid[i] / 2)
        for i in eachindex(φgrid)
    ]

    if φ ≤ φgrid[2]
        return Dgrid[2]
    elseif φ ≥ φgrid[end - 1]
        return Dgrid[n - 1]
    end

    rightindex = searchsortedfirst(φgrid, φ)
    leftindex = rightindex - 1

    weight = (φ - φgrid[leftindex]) / (φgrid[rightindex] - φgrid[leftindex])
    return (1 - weight) * Dgrid[leftindex] + weight * Dgrid[rightindex]
end

"Computes the optimal tax `τ` t time `t` and belief `φ`."
function dynamictax(t, φ, dynamicparameters)
    _, _, _, tgrid, τᶜtraj, signal, government, firm = dynamicparameters
    τᶜ = FastInterpolations.linear_interp(tgrid, τᶜtraj, t; extrap = clampextrap)
    D = valuedifferential(t, φ, dynamicparameters)

    return ηᵈ(t, φ, D, τᶜ, signal, government, firm) * τᶜ
end

"Computes the optimal abatement `a` at time `t` and belief `φ`."
function dynamicabatement(t, φ, dynamicparameters)
    _, _, _, tgrid, τᶜtraj, _, _, firm = dynamicparameters
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
    abatementfigure = Plots.plot(xlims = (0, 1), xlabel = L"\phi", ylabel = L"a", legendtitle = L"t")

    for (i, t) in enumerate(tsamples)
        τᶜ = FastInterpolations.linear_interp(tgrid, τᶜtraj, t; extrap = clampextrap)
        
        Plots.hline!([aᶜ(t, τᶜ, firm)]; c = timecolors[i], linestyle = :dash, label = false)
        Plots.plot!(abatementfigure, φgrid, φ -> dynamicabatement(t, φ, dynamicparameters); label = t, c = timecolors[i])
    end

    abatementfigure
end


## Define belief dynamics function
function precision(t, φ, dynamicparameters)
    _, _, _, tgrid, τᶜtraj, signal, _, _ = dynamicparameters
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
    _, _, _, tgrid, τᶜtraj, _, _, firm = dynamicparameters
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
