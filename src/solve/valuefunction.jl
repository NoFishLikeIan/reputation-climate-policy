using BoundaryValueDiffEq: MIRK4, TwoPointBVProblem
using DifferentialEquations: solve

struct ReputationProblem{TF, TG, TS, TT, TP}
    firm::TF
    government::TG
    signal::TS
    taxc::TT
    phispan::TP
end

function ReputationProblem(;
    firm = Firm(),
    government = Government(),
    signal = Signal(),
    taxc = committedtax(government, firm),
    phiepsilon = 1e-2,
)
    phispan = (phiepsilon, 1 - phiepsilon)
    ReputationProblem(firm, government, signal, taxc, phispan)
end

function boundarycosts(problem::ReputationProblem)
    firm = problem.firm
    government = problem.government
    taxc = problem.taxc

    lowercost = welfare(zero(taxc), zero(taxc), government, firm)
    uppercost = committedwelfare(taxc, government, firm)

    lowercost, uppercost
end

function valueguess(belief, problem::ReputationProblem)
    government = problem.government
    philower, phiupper = problem.phispan
    lowercost, uppercost = boundarycosts(problem)

    valueslope = (uppercost - lowercost) / (phiupper - philower)
    value = lowercost + valueslope * (belief - philower)
    reputation = -valueslope * belief * (1 - belief) / government.r

    [value, reputation]
end

function valuedynamics!(dx, x, problem::ReputationProblem, belief)
    firm = problem.firm
    government = problem.government
    signal = problem.signal
    taxc = problem.taxc

    value, reputation = x
    abatement = equilibriumabatement(belief, reputation, signal, government, firm, taxc)
    tax = equilibriumtax(belief, reputation, signal, government, firm, taxc)
    beliefvariance = belief * (1 - belief)
    taxgap = max(taxc - tax, sqrt(eps(typeof(taxc))) * max(one(taxc), abs(taxc)))
    signalfactor = signal.sigma / (signal.epsilon * taxgap)

    dx[1] = -government.r * reputation / beliefvariance
    dx[2] = (reputation + 2 * signalfactor^2 * (welfare(tax, abatement, government, firm) - value)) / beliefvariance

    dx
end

function boundaryvalueproblem(problem::ReputationProblem)
    lowercost, uppercost = boundarycosts(problem)

    function leftboundary!(resid, x, problem)
        resid[1] = x[1] - lowercost
    end

    function rightboundary!(resid, x, problem)
        resid[1] = x[1] - uppercost
    end

    TwoPointBVProblem(
        valuedynamics!,
        (leftboundary!, rightboundary!),
        belief -> valueguess(belief, problem),
        problem.phispan,
        problem;
        bcresid_prototype = (zeros(1), zeros(1)),
    )
end

function solvevaluefunction(problem::ReputationProblem; dt = first(problem.phispan), abstol = 1e-8, reltol = 1e-8)
    solve(boundaryvalueproblem(problem), MIRK4(); dt, abstol, reltol)
end
