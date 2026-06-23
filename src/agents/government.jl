Base.@kwdef struct Government{T <: Real}
    y₀::T = y₀
    r::T = 1e-2
end

function calibratedδ(loss, exposure)
    if exposure ≤ 0
        return zero(exposure)
    end
    
    return 2loss / exposure^2
end

function residualexposure(a, τ, firm::Firm)
    τ * √(firm.e₀ * e(a, firm))
end

function retirementexposure(a, τ, firm::Firm)
    τ * (a - firm.ω * firm.e₀)
end

function residualρ(a, τ, government::Government, firm::Firm)
    residualexposure(a, τ, firm) / government.y₀
end

function retirementρ(a, τ, government::Government, firm::Firm)
    retirementexposure(a, τ, firm) / government.y₀
end

function residualδ(government::Government, firm::Firm)
    calibratedδ(firm.lresidual₀, residualρ(firm.a₀, τ₀, government, firm))
end

function retirementδ(government::Government, firm::Firm)
    calibratedδ(firm.lretirement₀, retirementρ(firm.e₀, netzeroτ(government, firm), government, firm))
end

function residualshare(δ, government::Government, firm::Firm)
    δ * residualρ(firm.a₀, τ₀, government, firm)^2 / 2
end

function retirementshare(δ, government::Government, firm::Firm)
    δ * retirementρ(firm.e₀, netzeroτ(government, firm), government, firm)^2 / 2
end

function δᵣ(government::Government, firm::Firm)
   residualδ(government, firm) / 2government.y₀ 
end

function δₐ(government::Government, firm::Firm)
    retirementδ(government, firm) / 2government.y₀
end

function R(a, government::Government, firm::Firm)
    δᵣ(government, firm) * firm.e₀ * e(a, firm)
end

function A(a, government::Government, firm::Firm)
    δₐ(government, firm) * max(a - firm.ω * firm.e₀, 0)^2
end

function l(τ, a, government::Government, firm::Firm)
    τ^2 * (R(a, government, firm) + A(a, government, firm)) / 2
end

function w(m, τ, a, climate::Climate, government::Government, firm::Firm)
    government.y₀ * ( d(m, climate) + c(a, firm) ) + l(τ, a, government, firm)
end
