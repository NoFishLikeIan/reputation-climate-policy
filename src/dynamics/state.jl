function cumulativeemissionsdrift(a, firm::Firm)
    e(a, firm)
end

function abatementdrift(u)
    u
end

function investmentratedrift(a, u, τ, firm::Firm)
    firm.r * u + (firm.r * c(a, firm) - τ) / firm.ξ
end

"Carbon tax that keeps the abatement level stationary when the investment rate is zero."
function sustainingtax(a, firm::Firm)
    firm.r * c(a, firm)
end
