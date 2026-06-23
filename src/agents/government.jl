Base.@kwdef struct Government{T <: Real}
    y₀::T = y₀
    r::T = 1e-2
end

function calibratedδ(loss, exposure)
    exposure ≤ 0 && return zero(exposure)
    return 2loss / exposure^2
end

function residualexposure(a, τ, firm::Firm)
    residualemissions = max(e(a, firm), zero(a))
    return τ * √(firm.e₀ * residualemissions)
end

function retirementexposure(a, τ, firm::Firm)
    excessabatement = max(a - firm.ω * firm.e₀, zero(a))
    return τ * excessabatement
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

function residualloss(a, τ, government::Government, firm::Firm)
    residualemissions = max(e(a, firm), zero(a))
    return residualδ(government, firm) * τ^2 * firm.e₀ * residualemissions / (2government.y₀)
end

function retirementloss(a, τ, government::Government, firm::Firm)
    excessabatement = max(a - firm.ω * firm.e₀, zero(a))
    return retirementδ(government, firm) * τ^2 * excessabatement^2 / (2government.y₀)
end

function l(a, τ, government::Government, firm::Firm)
    residualloss(a, τ, government, firm) + retirementloss(a, τ, government, firm)
end

function w(m, τ, a, climate::Climate, government::Government, firm::Firm)
    government.y₀ * ( d(m, climate) + c(a, firm) ) + l(a, τ, government, firm)
end
