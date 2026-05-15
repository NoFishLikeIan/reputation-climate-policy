import Optim

function committedwelfare(tax, government::Government, firm::Firm)
    abatement = committedabatement(tax, firm)
    welfare(tax, abatement, government, firm)
end

function committedtax(government::Government, firm::Firm)
    upperbound = firm.nu * firm.e0
    result = Optim.optimize(tax -> committedwelfare(tax, government, firm), zero(upperbound), upperbound)

    Optim.minimizer(result)
end

function lagrangian(tax, abatement, reputation, taxc, signal::Signal, government::Government, firm::Firm)
    reputationvalue =
        reputation *
        signaldrift(tax, signal) *
        signalgap(tax, taxc, signal) /
        signal.sigma^2

    welfare(tax, abatement, government, firm) - reputationvalue
end

function b(reputation, signal::Signal, government::Government, firm::Firm)
    if reputation <= zero(reputation)
        return oftype(reputation, Inf)
    end

    government.delta * firm.e0 / (government.y0 * reputation * (signal.epsilon / signal.sigma)^2)
end

function taxshare(abatement, reputation, signal::Signal, government::Government, firm::Firm)
    burden = b(reputation, signal, government, firm)

    if !isfinite(burden)
        return zero(abatement + reputation)
    end

    1 / (2 + burden * emissions(abatement, firm))
end

function taxshare(belief, reputation, signal::Signal, government::Government, firm::Firm, taxc)
    maxabatement = taxc * (belief + (1 - belief) / 2) / firm.nu

    if maxabatement >= firm.e0
        return one(taxc) / 2
    end

    burden = b(reputation, signal, government, firm)

    if !isfinite(burden)
        return zero(taxc)
    elseif burden <= zero(burden)
        return one(taxc) / 2
    end

    residual = firm.e0 - taxc * belief / firm.nu
    linearpart = 2 + burden * residual
    discriminant = linearpart^2 - 4 * burden * (1 - belief) * taxc / firm.nu

    2 / (linearpart + sqrt(max(discriminant, zero(discriminant))))
end

function equilibriumtax(belief, reputation, signal::Signal, government::Government, firm::Firm, taxc)
    taxshare(belief, reputation, signal, government, firm, taxc) * taxc
end

function equilibriumabatement(belief, reputation, signal::Signal, government::Government, firm::Firm, taxc)
    tax = equilibriumtax(belief, reputation, signal, government, firm, taxc)
    bestresponseabatement(tax, belief, taxc, firm)
end

function equilibriumwelfare(belief, reputation, signal::Signal, government::Government, firm::Firm, taxc)
    tax = equilibriumtax(belief, reputation, signal, government, firm, taxc)
    abatement = bestresponseabatement(tax, belief, taxc, firm)

    welfare(tax, abatement, government, firm)
end
