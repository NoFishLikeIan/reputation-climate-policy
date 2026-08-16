function parameterstring(x)
    # `string` uses Julia's shortest round-trippable representation for floats,
    # so distinct parameter values are not collapsed by display rounding.
    replace(string(x), "+" => "")
end

function dynamicsolutionlabel(firm::Firm)
    join((
        "e0$(parameterstring(firm.e₀))",
        "a0$(parameterstring(firm.a₀))",
        "kappa$(parameterstring(firm.κ))",
        "xi$(parameterstring(firm.ξ))",
        "firmdiscount$(parameterstring(firm.r))",
    ), "_")
end

function solutionlabel(climate::Climate, government::Government, firm::Firm)
    join((
        dynamicsolutionlabel(firm),
        "y0$(parameterstring(government.y₀))",
        "r$(parameterstring(government.r))",
        "delta$(parameterstring(government.δ))",
        "gamma$(parameterstring(climate.γ))",
        "zeta$(parameterstring(climate.ζ))",
        "m0$(parameterstring(climate.m₀))",
    ), "_")
end

function solutionlabel(climate::Climate, government::Government, firm::Firm, signal::Signal)
    join((
        solutionlabel(climate, government, firm),
        "epsilon$(parameterstring(signal.ϵ))",
        "sigma$(parameterstring(signal.σ))",
    ), "_")
end
