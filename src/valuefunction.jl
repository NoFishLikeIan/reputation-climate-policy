function instantenouswelfare(τ, (a, p), V::FirmValue{T}, firm::Firm, government::Government, stategrid::Grid{2, T}, controlgrid::Grid{2, T}) where T
    abatements, beliefs = stategrid.ranges
    taxes = controlgrid.ranges[2]

    φ = interpolate(V.investment, (a, p, τ), (abatements, beliefs, taxes))
    a′ = f(φ, a, firm)

    damages = government.y₀ * government.β * d(e(a′, firm), government)

    welfare = c(φ, firm) + damages 

    return a′, welfare
end

function governmentstep!(S::Welfare{T}, V::FirmValue{T}, firm::Firm, government::Government, stategrid::Grid{2, T}, controlgrid::Grid{2, T}) where T
    abatements, beliefs = stategrid.ranges
    τₗ, τₐ = controlgrid.domains[2]

    @inbounds for (j, p) in enumerate(beliefs), (i, a) in enumerate(abatements)
        # Committed
        aᶜ′, welfareᶜ = instantenouswelfare(τᶜ, (a, p), V, firm, government, stategrid, controlgrid)
        xᶜ′ = (aᶜ′, p)
        Sᶜ′ = interpolate(S.welfare, xᶜ′, stategrid.ranges)
        committedwelfare = welfareᶜ + government.β * Sᶜ′

        # Deviate
        revealobjective = @closure τ -> begin
            a′, welfare = instantenouswelfare(τ, (a, p), V, firm, government, stategrid, controlgrid)
            x′ = (a′, zero(T))
            S′ = interpolate(S.welfare, x′, stategrid.ranges)

            return welfare + government.β * S′
        end
        
        res = Optim.optimize(revealobjective, τₗ, τₐ)
        uncommittedwelfare = Optim.minimum(res)
        τ = Optim.minimizer(res)
        
        shouldreveal = uncommittedwelfare < committedwelfare

        S.tax[i, j] = ifelse(shouldreveal, τ, τᶜ)
        S.welfare[i, j] = ifelse(shouldreveal, uncommittedwelfare, committedwelfare)
    end
end

function firmstep!(V::FirmValue{T}, S::Welfare{T}, firm::Firm, government::Government, stategrid::Grid{2, T}, controlgrid::Grid{2, T}) where T
    abatements, beliefs = stategrid.ranges
    _, taxes = controlgrid.ranges

    for (k, τ) in enumerate(taxes)
        for (j, p) in enumerate(beliefs), (i, a) in enumerate(abatements)

            obj = @closure φ -> begin

                a′ = f(φ, a, firm)

                # FIXME: Find the fixed point of beliefs.
                # Are beliefs consistent with a committed policy?
                τᶜ′ = interpolate(S.tax, (a′, p), stategrid.ranges)
                iscommconsistent =  τᶜ′ ≈ τᶜ
                
                # Deviation is always consistent
                τᵈ′ = interpolate(S.tax, (a′, zero(a)), stategrid.ranges)


                Vᵉ = (1 - p₁) * V₂itp(a₂, p₁, τ₃itp(a₁, p₁)) + p₁ * V₂itp(a₂, p₁, τᶜ)
                return c(φ, firm) - firm.β * a₂ * τ₂ + firm.β * Vᵉ
            end
            

        end
    end

end
