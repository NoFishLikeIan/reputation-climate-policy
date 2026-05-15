Base.@kwdef struct Government{T <: Real}
    xi::T = defaultscc / (defaulte0 * ctoCO2)
    y0::T = defaulty0
    delta::T = 10.0
    r::T = 0.01
end

function damages(emissions, government::Government)
    government.xi * emissions^2 / 2
end

function transitionliability(abatement, tax, firm::Firm)
    tax * sqrt(firm.e0 * emissions(abatement, firm))
end

function liabilityshare(abatement, tax, government::Government, firm::Firm)
    transitionliability(abatement, tax, firm) / government.y0
end

function transitionloss(abatement, tax, government::Government, firm::Firm)
    government.delta * tax^2 * firm.e0 * emissions(abatement, firm) / (2 * government.y0)
end

function welfare(tax, abatement, government::Government, firm::Firm)
    government.y0 * damages(emissions(abatement, firm), government) +
        abatementcost(abatement, firm) +
        transitionloss(abatement, tax, government, firm)
end
