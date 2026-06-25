function wᶜ(m, τᶜ, climate::Climate, government::Government, firm::Firm)
    w(m, τᶜ, a(τᶜ, government, firm), climate, government, firm)
end

function optimalcommittedtax(∂ₘu, government::Government{T}, firm::Firm{T}) where T

    if ∂ₘu ≤ 0
        return zero(T)
    end

    maxτ = netzeroτ(government, firm)
    
    obj = @closure τ -> begin
        aᶜ = a(τ, government, firm)
        return government.r * (government.y₀ * c(aᶜ, firm) + l(τ, aᶜ, government, firm)) - aᶜ * ∂ₘu
    end

    result = Optim.optimize(obj, 0., maxτ, brent)
    
    if !Optim.converged(result)
        return T(NaN)
    else
        return Optim.minimizer(result)
    end
end
