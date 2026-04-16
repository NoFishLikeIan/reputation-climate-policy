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
a₀ = 0.5
τᶜ = optimize(τᶜ -> w̄(a, τᶜ, firm, government, signal), 0., 1., Brent()).minimizer
ns = (101, 31, 15)
abatementspace = range(0, 1.25blissabatement(τᶜ, firm, signal), ns[1])
logitspace, _ = gausshermite(ns[2])
stategrid = Grid((abatementspace, logitspace))

firm = Firm()
government = Government()
signal = Signal(1., 1., gausshermite(ns[3]))


## Value Function
firmvalue = ValueFunction(stategrid, signal); firmvalue.V .= 0.
welfare = ValueFunction(stategrid); welfare.V .= d(e(0., firm), government) / (1 - government.β)

for (i, a) in enumerate(abatementspace)
	firmvalue.V[i, 1, :] .= 0.
	welfare.V[i, 1] = d(e(0., firm), government) / (1 - government.β)

	firmvalue.V[i, end, :] .= v̄(a, τᶜ, firm, signal)
	welfare.V[i, end] = w̄(a, τᶜ, firm, government, signal) 
end


## Iteration
innerparams = Dict(:maxiter => 100, :valtol => 1e-5, :poltol => 1e-3)

nestedpfi!(firmvalue, welfare, τᶜ, signal, stategrid, firm, government; maxiter = 50, valtol = 1e-5, poltol = 1e-5, verbose = 2, firmparams = innerparams, welfareparams = innerparams)
