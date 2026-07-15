function construct∂ᵐ(mgrid)
    n = length(mgrid)

    Δm = -diff(mgrid)
    rates = inv.(Δm)

    lower = rates
    diagonal = -vcat(first(rates), rates)
    upper = zeros(eltype(rates), n - 1); upper[1] = first(rates)

    return SparseArrays.spdiagm(-1 => lower, 0 => diagonal, 1 => upper)
end

function solvelcp(∂ₘ, a, τᶜ::TP, firm::Firm) where TP <: AbstractArray
    Bₘ = firm.r * LinearAlgebra.I - ∂ₘ
    x = firm.r .* τᶜ .- firm.r^2 * c(a, firm) # r (τᶜ - B * c(a))

    problem = LCPsolve.LCP(Bₘ, x)
    result = LCPsolve.solve!(problem)

    if !result.converged
        @warn "Solver did not converge with a = $a"
    end

    return result.sol
end

function solvefirmproblem!(q, I, τᶜ, collocationpoints, firm::Firm)
    agrid = getindex.(collocationpoints[:, 1], 1)
    mgrid = getindex.(collocationpoints[1, :], 2)
    ∂ᵐ = construct∂ᵐ(mgrid)
    
    @inbounds Threads.@threads for i in eachindex(agrid)
        taxpointsᵢ = @view collocationpoints[i, :]
        τᶜᵢ = τᶜ.(taxpointsᵢ)
        taxscale = max(maximum(abs, τᶜᵢ), one(eltype(τᶜᵢ)))
        taxtolerance = sqrt(eps(eltype(τᶜᵢ))) * taxscale
        taxminimum = minimum(τᶜᵢ)
        taxminimum ≥ -taxtolerance || throw(
            DomainError(taxminimum, "the committed tax must be non-negative")
        )
        τᶜᵢ .= max.(τᶜᵢ, zero(eltype(τᶜᵢ)))
        aᵢ = agrid[i]
        ∂ₘᵢ = e(aᵢ, firm) .* ∂ᵐ

        zᵢ = solvelcp(∂ₘᵢ, aᵢ, τᶜᵢ, firm)
        xᵢ = firm.r * c(aᵢ, firm)

        q[i, :] .= xᵢ .- zᵢ
        I[i] = findlast(≥(xᵢ), @view(q[i, :]))
    end

    return q, I
end
