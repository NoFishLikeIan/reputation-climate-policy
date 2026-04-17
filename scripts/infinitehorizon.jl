## Modules
using Revise
using UnPack
using Plots

using FastInterpolations
using FastGaussQuadrature
using LogExpFunctions
using FastClosures
using Optim

using Printf

default(label = false, dpi = 180)

## Imports
includet("../src/constants.jl")
includet("../src/signal.jl")
includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")

includet("../src/grid.jl")
includet("../src/valuefunction.jl")
includet("../src/boundary.jl")
includet("../src/pfi.jl")

## Setup
firm = Firm()
government = Government()

ns = (51, 31, 15)
signal = Signal(1., 1., gausshermite(ns[3]))

a₀ = 0.5
τᶜ = optimize(τᶜ -> w̄(a₀, τᶜ, firm, government, signal), 0., 1., Brent()).minimizer
abatementspace = range(0, 1.25blissabatement(τᶜ, firm, signal), ns[1])
logitspace, _ = gausshermite(ns[2])
stategrid = Grid((abatementspace, logitspace))

## Value Function
firmvalue = ValueFunction(stategrid, signal); firmvalue.V .= 0.
firmvalue.P .= φ̄(τᶜ, firm, signal)
welfare = ValueFunction(stategrid); welfare.V .= d(e(0., firm), government) / (1 - government.β)
welfare.P .= τ̄(τᶜ, firm, government)

## Iteration
innerparams = Dict(:maxiter => 100, :valtol => 1e-5, :poltol => 1e-3)

nestedpfi!(firmvalue, welfare, τᶜ, signal, stategrid, firm, government; maxiter = 50, valtol = 1e-5, poltol = 1e-5, verbose = 2, firmparams = innerparams, welfareparams = innerparams)
