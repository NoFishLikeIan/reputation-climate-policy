using Revise
using BenchmarkTools
using FastClosures
using Base.Threads
using UnPack

using Optim, Roots
using Interpolations

using Plots, LaTeXStrings, Printf

Plots.default(linewidth = 2, dpi = 180, label = false, background = :white)
plotpath = "figures/preliminaries"; if !ispath(plotpath) mkpath(plotpath) end

includet("../src/constants.jl")
includet("../src/agents/firm.jl")
includet("../src/agents/government.jl")
includet("../src/grid.jl")
includet("../src/optimal.jl")

firm = Firm()
government = Government()

abatementdomain = (0., 1.2) .* firm.e₀
beliefdomain = (0., 1.)
stategrid = Grid((51, 50), (abatementdomain, beliefdomain));

investmentspace = (0., 0.1) .* firm.e₀
taxspace = (0., 20.)
controlgrid = Grid((61, 60), (investmentspace, taxspace));

const τᶜ = 10.

# t = 2
## Firm
V₂ = FirmValue(Float64, stategrid, controlgrid)
for idx in CartesianIndices(V₂.value)
    τ = controlgrid.ranges[2][idx[3]]
    p = stategrid.ranges[2][idx[2]]
    a = stategrid.ranges[1][idx[1]]
    
    
    res = Optim.optimize(φ -> c(φ, firm) - firm.β * f(φ, a, firm) * τ, controlgrid.domains[1]...)
    
    V₂.value[idx] = Optim.minimum(res)
    V₂.investment[idx] = Optim.minimizer(res)
end

## Government
φ₁itp = extrapolate(interpolate((stategrid.ranges..., controlgrid.ranges[2]), V₂.investment, Gridded(Linear())), Interpolations.Line())
S₂ = Welfare(Float64, stategrid)
for jdx in CartesianIndices(S₂.welfare)
    p = stategrid.ranges[2][jdx[2]]
    a = stategrid.ranges[1][jdx[1]]

    obj = @closure τ -> begin
        φ = φ₁itp(a, p, τ)
        a′ = f(φ, a, firm)
        return c(φ, firm) + government.y₀ * (d(e(a, firm), government) + government.β * d(e(a′, firm), government))
    end
        
    res = Optim.optimize(obj, controlgrid.domains[2]...)

    S₂.welfare[jdx] = Optim.minimum(res)
    S₂.tax[jdx] = Optim.minimizer(res)
end

# t = 1
## Firm
V₂itp = extrapolate(interpolate((stategrid.ranges..., controlgrid.ranges[2]), V₂.value, Gridded(Linear())), Interpolations.Line())
τ₃itp = extrapolate(interpolate(stategrid.ranges, S₂.tax, Gridded(Linear())), Interpolations.Line())

V₁ = copy(V₂)
for idx in CartesianIndices(V₁.value)
    τ₂ = controlgrid.ranges[2][idx[3]]
    p₁ = stategrid.ranges[2][idx[2]]
    a₁ = stategrid.ranges[1][idx[1]]
    
    obj = @closure φ -> begin
        a₂ = f(φ, a₁, firm)
        # FIXME: This assumes constant beliefs
        Vᵉ = p₁ * V₂itp(a₂, p₁, τ₃itp(a₁, p₁)) + (1 - p₁) * V₂itp(a₂, p₁, τᶜ)
        return c(φ, firm) - firm.β * a₂ * τ₂ + firm.β * Vᵉ
    end
    
    res = Optim.optimize(obj, controlgrid.domains[1]...)
    
    V₁.value[idx] = Optim.minimum(res)
    V₁.investment[idx] = Optim.minimizer(res)
end

## Government
S₁ = copy(S₂)
φ₁itp = extrapolate(interpolate((stategrid.ranges..., controlgrid.ranges[2]), V₁.investment, Gridded(Linear())), Interpolations.Line())
S₂itp = extrapolate(interpolate(stategrid.ranges, S₂.welfare, Gridded(Linear())), Interpolations.Line())
for jdx in CartesianIndices(S₁.welfare)
    p₁ = stategrid.ranges[2][jdx[2]]
    a₁ = stategrid.ranges[1][jdx[1]]

    obj = @closure τ₂ -> begin
        φ₁ = φ₁itp(a₁, p₁, τ₂)
        a₂ = f(φ₁, a₁, firm)

        damages = d(e(a₁, firm), government) + government.β * d(e(a₂, firm), government)

        p₂ = (τ₂ ≈ τᶜ) ? p₁ : 0.

        return c(φ₁, firm) + government.y₀ * damages + government.β * S₂itp(a₂, p₂)
    end
        
    res = Optim.optimize(obj, controlgrid.domains[2]...)

    S₁.welfare[jdx] = Optim.minimum(res)
    S₁.tax[jdx] = Optim.minimizer(res)
end

# t = 0
## Firm 
# FIXME: I think the firm should take into account the τ policy.
V₁itp = extrapolate(interpolate((stategrid.ranges..., controlgrid.ranges[2]), V₂.value, Gridded(Linear())), Interpolations.Line())
τ₂itp = extrapolate(interpolate(stategrid.ranges, S₁.tax, Gridded(Linear())), Interpolations.Line())

V₀ = copy(V₁)
for idx in CartesianIndices(V₀.value)
    τ₁ = controlgrid.ranges[2][idx[3]]
    p₀ = stategrid.ranges[2][idx[2]]
    a₀ = stategrid.ranges[1][idx[1]]
    
    obj = @closure φ -> begin
        a₁ = f(φ, a₀, firm)
        Vᵉ = p₀ * V₂itp(a₁, p₀, τ₃itp(a₁, p₀)) + (1 - p₀) * V₂itp(a₁, p₀, τᶜ)
        return c(φ, firm) - firm.β * a₁ * τ₁ + firm.β * Vᵉ
    end
    
    res = Optim.optimize(obj, controlgrid.domains[1]...)
    
    V₀.value[idx] = Optim.minimum(res)
    V₀.investment[idx] = Optim.minimizer(res)
end

## Government
S₀ = copy(S₁)
φ₀itp = extrapolate(interpolate((stategrid.ranges..., controlgrid.ranges[2]), V₀.investment, Gridded(Linear())), Interpolations.Line())
S₁itp = extrapolate(interpolate(stategrid.ranges, S₁.welfare, Gridded(Linear())), Interpolations.Line())

for jdx in CartesianIndices(S₀.welfare)
    p₀ = stategrid.ranges[2][jdx[2]]
    a₀ = stategrid.ranges[1][jdx[1]]

    obj = @closure τ₂ -> begin
        φ₁ = φ₁itp(a₀, p₀, τ₂)
        a₁ = f(φ₁, a₀, firm)

        damages = d(e(a₀, firm), government) + 
            government.β * d(e(a₁, firm), government)

        p₁ = (τ₂ ≈ τᶜ) ? p₀ : 0.

        return c(φ₁, firm) + government.y₀ * damages + government.β * S₁itp(a₁, p₁)
    end
        
    res = Optim.optimize(obj, controlgrid.domains[2]...)

    S₁.welfare[jdx] = Optim.minimum(res)
    S₁.tax[jdx] = Optim.minimizer(res)
end

## Firm
S₀ = copy(S₁)
φ₀itp = extrapolate(interpolate((stategrid.ranges..., controlgrid.ranges[2]), V₀.investment, Gridded(Linear())), Interpolations.Line())
S₁itp = extrapolate(interpolate(stategrid.ranges, S₁.welfare, Gridded(Linear())), Interpolations.Line())

for jdx in CartesianIndices(S₀.welfare)
    p₀ = stategrid.ranges[2][jdx[2]]
    a₀ = stategrid.ranges[1][jdx[1]]

    obj = @closure τ₂ -> begin
        φ₁ = φ₁itp(a₀, p₀, τ₂)
        a₁ = f(φ₁, a₀, firm)

        damages = d(e(a₀, firm), government) + 
            government.β * d(e(a₁, firm), government)

        p₁ = (τ₂ ≈ τᶜ) ? p₀ : 0.

        return c(φ₁, firm) + government.y₀ * damages + government.β * S₁itp(a₁, p₁)
    end
        
    res = Optim.optimize(obj, controlgrid.domains[2]...)

    S₁.welfare[jdx] = Optim.minimum(res)
    S₁.tax[jdx] = Optim.minimizer(res)
end

## Simulation
a₀ = 0.
p₀ = rand()

τ₁ = interpolate(stategrid.ranges, S₀.tax, Gridded(Linear()))(a₀, p₀)
