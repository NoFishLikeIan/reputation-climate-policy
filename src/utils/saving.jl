import SHA

function parameterstring(x)
    # `string` uses Julia's shortest round-trippable representation for floats,
    # so distinct parameter values are not collapsed by display rounding.
    replace(string(x), "+" => "")
end

function dynamicsolutionlabel(household::Household, firm::Firm)
    join((
        "nu$(parameterstring(household.ν))",
        "varphiL$(parameterstring(household.φᴸ))",
        "householddiscount$(parameterstring(household.r))",
        "e0$(parameterstring(firm.e₀))",
        "a0$(parameterstring(firm.a₀))",
        "A$(parameterstring(firm.A))",
        "etaE$(parameterstring(firm.ηᴱ))",
        "kappa$(parameterstring(firm.κ))",
        "xi$(parameterstring(firm.ξ))",
        "firmdiscount$(parameterstring(firm.r))",
    ), "_")
end

function solutionlabel(household::Household, firm::Firm, government::Government, climate::Climate)
    join((
        dynamicsolutionlabel(household, firm),
        "y0$(parameterstring(government.y₀))",
        "r$(parameterstring(government.r))",
        "delta$(parameterstring(government.δ))",
        "gamma$(parameterstring(climate.γ))",
        "zeta$(parameterstring(climate.ζ))",
        "m0$(parameterstring(climate.m₀))",
    ), "_")
end

function signallabel(signal::Signal)
    join((
        "epsilon$(parameterstring(signal.ϵ))",
        "sigma$(parameterstring(signal.σ))",
    ), "_")
end

function solutionlabel(household::Household, firm::Firm, government::Government, signal::Signal, climate::Climate)
    join((
        solutionlabel(household, firm, government, climate),
        signallabel(signal),
    ), "_")
end

function taxmethodlabel(taxmethod)
    fields = (
        "$(field)$(parameterstring(getfield(taxmethod, field)))"
        for field in fieldnames(typeof(taxmethod))
    )

    join((string(nameof(typeof(taxmethod))), fields...), "_")
end

function solutionlabel(household::Household, firm::Firm, government::Government, signal::Signal, climate::Climate, taxmethod)
    join((
        solutionlabel(household, firm, government, signal, climate),
        "taxmethod$(taxmethodlabel(taxmethod))",
    ), "_")
end

"Short, stable filename determined by every committed-solution parameter."
function solutionfilename(household::Household, firm::Firm, government::Government, climate::Climate)
    digest = bytes2hex(SHA.sha256(solutionlabel(household, firm, government, climate)))
    "solution-$digest.jld2"
end

"JLD2 group containing one signal and tax-method-specific uncommitted solution."
function uncommittedsolutionkey(signal::Signal, taxmethod)
    join(("uncommitted", signallabel(signal), taxmethodlabel(taxmethod)), "/")
end
