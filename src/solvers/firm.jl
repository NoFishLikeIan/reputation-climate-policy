function crudebackwardstep!(Vₜ, Φₜ, τₜ, Vₜ₊₁, A, firm::Firm)
    @inbounds @threads for i in eachindex(Vₜ)
        aᵢ = A[i]
        ϕ = Φₜ[i]
        Vₜ[i] = c(aᵢ, ϕ, firm) + τₜ * emissions(aᵢ, firm) + firm.β * interpolate(Vₜ₊₁, f(aᵢ, ϕ, firm), A)
    end
end

function backwardstep!(Vₜ, Φₜ, τₜ, Vₜ₊₁, A, firm::Firm)
    @inbounds @threads for i in eachindex(Vₜ)
        aᵢ = A[i]
        
        vₜ, ϕₜ = gssmin(ϕ -> c(aᵢ, ϕ, firm) + τₜ * emissions(aᵢ, firm) + firm.β * interpolate(Vₜ₊₁, f(aᵢ, ϕ, firm), A), 0., 1.; tol = 1e-4)
        
        Vₜ[i] = vₜ
        Φₜ[i] = ϕₜ
    end
end

function steadystate!(valuefunction::FirmValue, τ, A, firm::Firm; kwargs...)
    V̄ = @view valuefunction.V[:, end]
    Φ̄ = @view valuefunction.Φ[:, end]

    steadystate!(V̄, Φ̄ , τ, A, firm; kwargs...)

    return valuefunction
end
function steadystate!(V::AbstractVector, Φ::AbstractVector, τ, A, firm::Firm; optimisationstep = 100, iterations = 2_000)
    for iter in 0:iterations
        if iter % optimisationstep > 0
            crudebackwardstep!(V, Φ, τ, V, A, firm)
        else
            backwardstep!(V, Φ, τ, V, A, firm)
        end
    end

    return V, Φ
end

function backwardinduction!(V::AbstractMatrix, Φ::AbstractMatrix, τ₀, θ, A, firm::Firm)
    T = size(V, 2)
    
    @inbounds for t in reverse(1:(T - 1))
        Vₜ = @view V[:, t]
        Vₜ₊₁ = @view V[:, t + 1]
        Φₜ = @view Φ[:, t]
        τₜ = τ(t, τ₀, θ)
        
        backwardstep!(Vₜ, Φₜ, τₜ, Vₜ₊₁, A, firm)
    end
    
    return V, Φ
end
function backwardinduction!(valuefunction::FirmValue, τ₀, θ, A, firm::Firm)
    backwardinduction!(valuefunction.V, valuefunction.Φ, τ₀, θ, A, firm)

    return valuefunction
end
