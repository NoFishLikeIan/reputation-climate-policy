using Revise
using BenchmarkTools
using FastClosures
using Base.Threads
using ForwardDiff, Roots

using Plots, LaTeXStrings, Printf

Plots.default(linewidth = 2, dpi = 180, label = false, background_color = :transparent)
plotpath = "figures/preliminaries"; if !ispath(plotpath) mkpath(plotpath) end

includet("../src/utils.jl")
includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")

includet("../src/solvers/firm.jl")
includet("../src/solvers/government.jl")

firm = Firm()
government = Government()

"Total price of investment paid by firm to obtain abatement `a`"
function totalprice(a, firm)
    firm.κ * a / firm.α
end

# Climate damages
A = range(0, 1, 101)

let
    xticks = 0:0.2:1
    xticklabels = [L"%$(100x)\%" for x in xticks]
    
    dfig = plot(; xlabel = L"Fraction of abated emissions $a$", ylabel = "tUSD / y", xticks = (xticks, xticklabels), xlims = (0, 1), margins = 2Plots.mm, ylims = (0, Inf))

    plot!(dfig, A, a -> d(a, firm, government) * government.y₀; c = :black, label = L"d(a) y_0")

    savefig(dfig, joinpath(plotpath, "damages.png"))

    dfig
end