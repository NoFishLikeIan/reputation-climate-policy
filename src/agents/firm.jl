Base.@kwdef struct Firm{T <: Real}
    e0::T = defaulte0
    nu::T = defaultdietzphi * defaulty0 * ctoCO2^2
end

emissions(abatement, firm::Firm) = firm.e0 - abatement

function abatementcost(abatement, firm::Firm)
    firm.nu * abatement^2 / 2
end

function firmcost(abatement, tax, firm::Firm)
    emissions(abatement, firm) * tax + abatementcost(abatement, firm)
end

function committedabatement(tax, firm::Firm)
    min(tax / firm.nu, firm.e0)
end

function expectedfirmcost(abatement, tax, belief, committedtax, firm::Firm)
    belief * firmcost(abatement, committedtax, firm) +
        (1 - belief) * firmcost(abatement, tax, firm)
end

function bestresponseabatement(tax, belief, committedtax, firm::Firm)
    expectedtax = belief * committedtax + (1 - belief) * tax
    min(expectedtax / firm.nu, firm.e0)
end
