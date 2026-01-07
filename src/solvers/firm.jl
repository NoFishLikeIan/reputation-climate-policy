function objective(ϕ, aᵢ, Vₜ₊₁, τₜ, A, firm::Firm)
    c(aᵢ, ϕ, firm) + τₜ * emissions(aᵢ, firm) + 
        firm.β * interpolate(Vₜ₊₁, f(aᵢ, ϕ, firm), A)
end

function optimalinvestment(aᵢ, Vₜ₊₁, τₜ, A, firm::Firm; ϕmin = 0., ϕmax = 100., tol = 1e-8)
    g = @closure ϕ -> objective(ϕ, aᵢ, Vₜ₊₁, τₜ, A, firm)
    return gssmin(g, ϕmin, ϕmax; tol = tol)
end

function crudebackwardstep!(Vₜ, Φₜ, τₜ, Vₜ₊₁, A, firm::Firm)
    @inbounds @threads for i in eachindex(Vₜ)
        aᵢ = A[i]
        ϕ = Φₜ[i]
        Vₜ[i] = objective(ϕ, aᵢ, Vₜ₊₁, τₜ, A, firm)
    end
end

function backwardstep!(Vₜ, Φₜ, τₜ, Vₜ₊₁, A, firm::Firm; optkwargs...)
    @inbounds @threads for i in eachindex(Vₜ)
        aᵢ = A[i]
        vₜ, ϕₜ = optimalinvestment(aᵢ, Vₜ₊₁, τₜ, A, firm; optkwargs...)
        
        Vₜ[i] = vₜ
        Φₜ[i] = ϕₜ
    end
end

function steadystate!(valuefunction::FirmValue, τ, A, firm::Firm; kwargs...)
    V = @view valuefunction.V[:, end]
    Φ = @view valuefunction.Φ[:, end]

    steadystate!(V, Φ, τ, A, firm; kwargs...)

    return valuefunction
end
function steadystate!(V::AbstractVector, Φ::AbstractVector, τ, A, firm::Firm; optimisationstep = 100, iterations = 2_000, optkwargs...)
    for iter in 0:iterations
        if iter % optimisationstep > 0
            crudebackwardstep!(V, Φ, τ, V, A, firm)
        else
            backwardstep!(V, Φ, τ, V, A, firm; optkwargs...)
        end
    end

    return V, Φ
end

function backwardinduction!(V::AbstractMatrix, Φ::AbstractMatrix, τ₀, θ, A, firm::Firm; optkwargs...)
    T = size(V, 2)
    
    @inbounds for t in reverse(1:(T - 1))
        Vₜ = @view V[:, t]
        Vₜ₊₁ = @view V[:, t + 1]
        Φₜ = @view Φ[:, t]
        τₜ = τ(t, τ₀, θ)
        
        backwardstep!(Vₜ, Φₜ, τₜ, Vₜ₊₁, A, firm, optkwargs...)
    end
    
    return V, Φ
end
function backwardinduction!(valuefunction::FirmValue, τ₀, θ, A, firm::Firm; kwargs...)
    backwardinduction!(valuefunction.V, valuefunction.Φ, τ₀, θ, A, firm; kwargs...)

    return valuefunction
end
