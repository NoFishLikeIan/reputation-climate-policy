using Revise
using BenchmarkTools
using FastClosures
using Base.Threads
using ForwardDiff, Roots

using Plots, LaTeXStrings, Printf

Plots.default(linewidth = 2, dpi = 180, label = false)
plotpath = "figures"; if !ispath(plotpath) mkpath(plotpath) end

includet("../src/utils.jl")
includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")

includet("../src/solvers/firm.jl")
includet("../src/solvers/government.jl")

firm = Firm()
government = Government()

n = 301
A = range(0, 1; length = n)

# Firm optimisation
T = 501
τ₀ = 3 * (3.67 * 1e9 * 1e-12)
θspace = range(0., 0.001; length = 3)

begin
    atrajectories = Dict{Float64, Vector{Float64}}()
    ϕtrajectories = Dict{Float64, Vector{Float64}}()
    investmentcosttrajectories = Dict{Float64, Vector{Float64}}()
    damagecosttrajectories = Dict{Float64, Vector{Float64}}()
    wtrajectories = Dict{Float64, Vector{Float64}}()
end

horizon = 80

for θ in θspace
	valuefunction = FirmValue(ones(n, T), ones(n, T) ./ 2);
    steadystate!(valuefunction, τ(T, τ₀, θ), A, firm; iterations = 10_000, optimisationstep = 1)

    let
        fig = plot(xlabel = L"Abatemnet level $a_t$", title = L"Tax level $\overline{\tau} = %$(τ(T, τ₀, θ))$")
        plot!(fig, A, valuefunction.V[:, end]; ylabel = L"Terminal costs $\overline{V} \; [\mathrm{tUSD}]$", yguidefontcolor = :darkblue, c = :darkblue, xlims = (0, 1), linestyle = :dash)
        plot!(twinx(fig), A, valuefunction.Φ[:, end]; ylabel = L"Investment in abatemnet $\overline{\phi}$", yguidefontcolor = :darkred, c = :darkred, xlims = (0, 1), ylims = (0., Inf))

        figpath = joinpath(plotpath, "committed", "steadystate"); if !ispath(figpath) mkpath(figpath) end
        filename = joinpath(figpath, "tau$(τ₀)_theta$(θ).png")
        savefig(fig, filename)

        fig
    end

    # Backard induction
    backwardinduction!(valuefunction, τ₀, θ, A, firm)
    let
        valuefig = contourf(1:T, A, valuefunction.V; xlabel = L"t", ylabel = L"a", title = L"Firm costs $V_t(a)$", camera = (15, 30), opacity = 1, xlims = (1, horizon), ylims = extrema(A), linewidth = 0.5, c = :viridis)

        investmentfig = contourf(1:T, A, valuefunction.Φ; xlabel = L"t", ylabel = L"a", title = L"Investment $\phi_t(a)$", camera = (15, 30), opacity = 1, xlims = (1, horizon), ylims = extrema(A), linewidth = 0.5, clims = (0, Inf), c = :Greens)

        jointfig = plot(valuefig, investmentfig; size = 500 .* (2√2, 1), margins = 10Plots.mm)

        figpath = joinpath(plotpath, "committed", "trajectory"); if !ispath(figpath) mkpath(figpath) end
        filename = joinpath(figpath, "tau$(τ₀)_theta$(θ).png")

        savefig(jointfig, filename)

        jointfig
    end

    # Path of variables	
	atrajectory = zeros(T)
	wtrajectory = zeros(T)
	ϕtrajectory = zeros(T - 1)
	investmentcosttrajectory = zeros(T - 1)
	damagecosttrajectory = zeros(T - 1)
	
	for t in 1:(T - 1)
		aₜ = atrajectory[t]
		ϕₜ = interpolate(valuefunction.Φ, (aₜ, t), (A, 1:T))
		investmentcost = c(aₜ, ϕₜ, firm)
		damagecost = d(aₜ, firm, government) * government.Y
		
		ϕtrajectory[t] = ϕₜ
		investmentcosttrajectory[t] = investmentcost
		damagecosttrajectory[t] = damagecost
		wtrajectory[t + 1] = wtrajectory[t] + socialcost(aₜ, ϕₜ, firm, government) * government.β^(t - 1)
		atrajectory[t + 1] = f(aₜ, ϕₜ, firm)
    end

	atrajectories[θ] = atrajectory
	ϕtrajectories[θ] = ϕtrajectory
	investmentcosttrajectories[θ] = investmentcosttrajectory
	damagecosttrajectories[θ] = damagecosttrajectory
	wtrajectories[θ] = wtrajectory
end

# Plot all trajectories together
let
    xticks = 0:10:horizon
    fig = plot(xlabel = "Year", margins = 5Plots.mm, ylabel = L"Fraction of abated emissions $a_t$", xlims = (1, horizon), xticks = (xticks, xticks .+ 2020))

    for (i, θ) in enumerate(θspace)
        plot!(fig, 1:T, atrajectories[θ]; label = L"$\theta = %$(round(θ, digits = 4))$")
    end
    figpath = joinpath(plotpath, "committed", "paths"); if !ispath(figpath) mkpath(figpath) end
    filename = joinpath(figpath, "tau$(τ₀)_atrajectories.png")
    savefig(fig, filename)
    fig
end

let
    fig = plot(xlabel = L"$t$", margins = 5Plots.mm, ylabel = L"Investment $\phi_t$", xlims = (1, horizon))

    for (i, θ) in enumerate(θspace)
        plot!(fig, 1:(T-1), ϕtrajectories[θ]; label = L"$\theta = %$(round(θ, digits = 5))$")
    end
    figpath = joinpath(plotpath, "committed", "paths"); if !ispath(figpath) mkpath(figpath) end
    filename = joinpath(figpath, "tau$(τ₀)_phistrajectories.png")
    savefig(fig, filename)
    fig
end

let
    fig = plot(xlabel = L"$t$", margins = 5Plots.mm, ylabel = "Investment costs - Damage costs", xlims = (1, horizon))
    for (i, θ) in enumerate(θspace)
        plot!(fig, 1:(T-1), investmentcosttrajectories[θ] - damagecosttrajectories[θ]; label = L"$\theta = %$(round(θ, digits = 5))$")
    end
    figpath = joinpath(plotpath, "committed", "paths"); if !ispath(figpath) mkpath(figpath) end
    filename = joinpath(figpath, "tau$(τ₀)_costdifferences.png")
    savefig(fig, filename)
    fig
end

let
    fig = plot(xlabel = L"$t$", margins = 5Plots.mm, ylabel = "Cumulative social costs", xlims = (1, horizon))
    for (i, θ) in enumerate(θspace)
        plot!(fig, 1:T, wtrajectories[θ]; label = L"$\theta = %$(round(θ, digits = 5))$")
    end
    figpath = joinpath(plotpath, "committed", "paths"); if !ispath(figpath) mkpath(figpath) end
    filename = joinpath(figpath, "tau$(τ₀)_cumulativewelfare.png")
    savefig(fig, filename)
    fig
end
