## New solver
abstract type AbstractModelParameters end

struct ModelParameters{H <: Household, F, G, C} <: AbstractModelParameters
    household::H
    firm::F
    government::G
    climate::C
end

"Model parameters specialised to linear labour supply, corresponding to `ϕ = 1`."
struct LinearLabourModelParameters{H <: LinearHousehold, F, G, C} <: AbstractModelParameters
    household::H
    firm::F
    government::G
    climate::C
end
function ModelParameters(household::H, firm::F, government::G, climate::C) where {H, F, G, C}
    isa(household, LinearHousehold) ? 
        LinearLabourModelParameters{H, F, G, C}(household, firm, government, climate) :
        ModelParameters{H, F, G, C}(household, firm, government, climate)
end


## Optimal tax
committedtaxresidual(τ, (x, model)) = committedtaxresidual(τ, x, model)
function committedtaxresidual(τ, x, model::AbstractModelParameters)
    @unpack household, firm, government, climate = model
    m, _, _, λₘ, _, λᵨ = x # State, co-state, cumulative costs (z, λ, u)

    wage = ω(τ, firm)

    return firm.A * n′(wage, household) * ω′(τ, firm) * (d(m, climate) + firm.η * ( λₘ - τ )) - firm.r * λᵨ
end
function committedtax(x::TX, model::AbstractModelParameters) where {T, TX <: AbstractVector{T}}
    𝟎 = zero(T)

    taxupperbound = (1 - √eps(T)) / model.firm.η
    foc = Base.Fix2(committedtaxresidual, (x, model))

    fl = foc(𝟎)
    fu = foc(taxupperbound)

    if fl ≥ 0
        return 𝟎
    elseif fu ≤ 0
        taxupperbound
    else
        return Roots.find_zero(foc, (𝟎, taxupperbound), Roots.Brent())
    end
end
function committedtax(x::TX, model::LinearLabourModelParameters) where {T, TX <: AbstractVector{T}}
    @unpack household, firm, climate = model
    m, _, _, λₘ, _, λᵨ = x # State, co-state, cumulative costs (z, λ, u)

    interiortax = (d(m, climate) / firm.η) + λₘ + firm.r * household.ν * λᵨ / (firm.A * firm.η)^2

    return clamp(interiortax, 0, inv(firm.η))
end

## State-Costate system
initialguess(model) = initialguess(SA.SVector, model)
function initialguess(TV, model::AbstractModelParameters)
    @unpack household, firm, government, climate = model

    τᶜ₀ = firm.r * c(firm.a₀, firm)
    λₘ₀ = τᶜ₀ - d(climate.m₀, climate) / firm.η    
    q₀ = 2 * firm.r * c(firm.a₀, firm)
    λₐ₀ = -q₀ / firm.r
    λᵨ₀ = zero(λₐ₀)

    return TV(climate.m₀, firm.a₀, q₀, λₘ₀, λₐ₀, λᵨ₀)
end

function driftcommitted!(dx, x, model::AbstractModelParameters, t)
    @unpack household, firm, government, climate = model 
    m, a, q, λₘ, λₐ, λᵨ = x # State, co-state, cumulative costs (z, λ, u)
    
    τᶜ = committedtax(x, model)
    wage = ω(τᶜ, firm)
    labour = n(wage, household)

    dm = firm.η * firm.A * labour - a
    da = e(labour, a, firm) > 0 ? α(q, a, firm) : zero(a)
    dq = firm.r * (q - τᶜ + firm.κ * da)

    dλₘ = government.r * λₘ - firm.A * labour * d′(m, climate)
    dλₐ = (government.r + firm.κ / firm.ξ) * λₐ + λₘ + (firm.r*firm.κ^2 / firm.ξ) * λᵨ + (a * firm.κ^2 / firm.ξ)
    dλᵨ = (government.r - firm.r - firm.κ / firm.ξ) * λᵨ - λₐ / (firm.r * firm.ξ) - q / (firm.r^2 * firm.ξ)

    # dw = exp(-government.r * t) * w(τᶜ, m, a, da, household, firm, government, climate)

    dx[1] = dm
    dx[2] = da
    dx[3] = dq
    dx[4] = dλₘ
    dx[5] = dλₐ
    dx[6] = dλᵨ
    # dx[7] = dw

    return dx
end

## Truncation
function ρd̄′(t, (τᶜ, x, model))
    m, a = @view x[1:2]
    wage = ω(τᶜ, model.firm)
    labour = n(wage, model.household)
    mₜ = m + e(labour, a, model.firm) * t

    return d′(mₜ, model.climate) * exp(-model.government.r * t)
end
function tρd̄′(t, (τᶜ, x, model))
    ρd̄′(t, (τᶜ, x, model)) * t
end

function ρd̄(t, (τᶜ, x, model))
    m, a = @view x[1:2]
    wage = ω(τᶜ, model.firm)
    labour = n(wage, model.household)
    mₜ = m + e(labour, a, model.firm) * t

    return d(mₜ, model.climate) * exp(-model.government.r * t)
end

"Terminal gradient of the value function"
function ∇v̄(x, model::AbstractModelParameters)
    @unpack household, firm, government, climate = model
    m, a, q = @view x[1:3]

    wage = ω(q, firm)

    output = firm.A * n(wage, household)
    output′ = firm.A * n′(wage, household) * ω′(q, firm)
    emissions = firm.η * output - a

    ecds = climate.γ * climate.ζ^2

    aₘ = ecds * m^2 / 2
    bₘ = government.r + m * ecds * emissions
    cₘ = ecds * emissions^2 / 2

    Jₘ = J(aₘ, bₘ, cₘ)
    Gₘ = G(aₘ, bₘ, cₘ)

    ∂ₘv = output * ecds * (m * Jₘ + emissions * Gₘ)
    ∂ₐv = -output * ecds * (m * Gₘ + emissions * L(aₘ, bₘ, cₘ))
    ∂ᵨv = output′ * ((1 / government.r) - Jₘ - firm.η * ∂ₐv - firm.η * q / government.r)

    return (∂ₘv, ∂ₐv, ∂ᵨv)
end


## Outer Optimization
struct CommittedPathParameters{TS <: Real, TP <: AbstractModelParameters}
    T::TS
    model::TP
end

function driftcommitted!(dx, x, p::CommittedPathParameters, t)
    driftcommitted!(dx, x, p.model, t)
end

optimisationbounds(model) = optimisationbounds(SA.SVector, model)
function optimisationbounds(TV, model::AbstractModelParameters; λmax = Inf)
    @unpack household, firm, government, climate = model
  
    q₀ = firm.r * c(firm.a₀, firm)

    lb = TV(climate.m₀, firm.a₀, q₀, -λmax, -λmax, -λmax)
    ub = TV(climate.m₀, firm.a₀, λmax, λmax, λmax, λmax)

    return lb, ub
end

function initialcondition(x₀::TX, p::CommittedPathParameters) where {T, TX <: AbstractVector{T}}
    initialcondition!(Vector{Float64}(undef, 3), x₀, p)
end
function initialcondition!(res, x₀, p::CommittedPathParameters)
    m, a, _, _, _, λᵨ = x₀ # State, co-state, cumulative costs (z, λ, u)₀

    res[1] = m - p.model.climate.m₀
    res[2] = a - p.model.firm.a₀
    res[3] = λᵨ

    return res
end

function terminalcondition(x̄::TX, p::CommittedPathParameters) where {T, TX <: AbstractVector{T}}
    terminalcondition!(Vector{Float64}(undef, 3), x̄, p)
end
function terminalcondition!(res, x̄, p::CommittedPathParameters)
    @unpack firm, household = p.model
    ∂ₘv, ∂ₐv, ∂ᵨv = ∇v̄(x̄, p.model)
    _, a, q, λₘ, λₐ, λᵨ = x̄ # State, co-state, cumulative costs (z, λ, u)̄

    wage = ω(q, firm)
    labour = n(wage, household)
    output′ = firm.η * firm.A * n′(wage, household) * ω′(q, firm)
    
    res[1] = e(labour, a, firm) # No emissions
    res[2] = ∂ₘv - λₘ
    res[3] = (λᵨ - ∂ᵨv) + output′ * (λₐ - ∂ₐv)

    return res
end

function solvecommittedpath(p::CommittedPathParameters; normalisedstep = 1e-3)
    # x₀ = initialguess(SA.MVector, p.model)
   
    # fn = BVP.BVPFunction(driftcommitted!, boundaryconditions!; bcresid_prototype = zeros(7))
    # problem = BVP.BVProblem(fn, x₀, (0., p.T), p)

    # dt = normalisedstep * p.S
    # solution = BVP.solve(problem, BVP.MIRK4(); dt = dt)
    

    
end

function e(x::TX, t, integrator::TI) where {TX <: AbstractVector, TI <: ODECore.ODEIntegrator}
    model = integrator.p.model
    
    τᶜ = committedtax(x, model) # FIXME: Inefficient to calculate twice if FOC is used
    wage = ω(τᶜ, model.firm)
    labour = n(wage, model.household)
    a, _ = x # State, co-state, cumulative costs (z, λ, u)[2]

    return e(labour, a, model.firm)
end
function isfullabatement(x, t, integrator)
    e(x, t, integrator) ≤ 0
end