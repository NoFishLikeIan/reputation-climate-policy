using Revise
using BenchmarkTools
using FastClosures
using Base.Threads
using ForwardDiff, Roots

using Plots, LaTeXStrings, Printf

Plots.default(linewidth = 2, dpi = 180, label = false)
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
    plot(A, a -> d(a, firm, government) * government.Y; xlabel = L"Fraction of abated emissions $a$", ylabel = "Climate damages [tUSD / y]")
    #plot!(A, a -> totalprice(a, firm))
end