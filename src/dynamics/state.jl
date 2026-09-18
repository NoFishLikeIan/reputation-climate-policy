function cumulativeemissionsdrift(τ, a, household::Household, firm::Firm)
    labour = n(τ, household, firm)
    return e(labour, a, firm)
end

function investmentratedrift(a, u, τ, firm::Firm)
    firm.r * u + (firm.r * c(a, firm) - τ) / firm.ξ
end

"Carbon tax that keeps the abatement level stationary when the investment rate is zero."
function sustainingtax(a, firm::Firm)
    firm.r * c(a, firm)
end
