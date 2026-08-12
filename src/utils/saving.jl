function parameterstring(x)
    replace(Printf.@sprintf("%.3e", x), "+" => "")
end

function dynamicsolutionlabel(firm)
    "omega$(parameterstring(firm.ω))_nu$(parameterstring(firm.ν))"
end

function solutionlabel(climate::Climate, government::Government, firm::Firm)
    join((
        "e0$(parameterstring(firm.e₀))",
        "kappa$(parameterstring(firm.κ))",
        "xi$(parameterstring(firm.ξ))",
        "firmdiscount$(parameterstring(firm.r))",
        "y0$(parameterstring(government.y₀))",
        "r$(parameterstring(government.r))",
        "gamma$(parameterstring(climate.γ))",
        "zeta$(parameterstring(climate.ζ))"
    ), "_")
end

function solutionlabel(climate::Climate, government::Government, firm::Firm, signal::Signal)
    join((
        solutionlabel(climate, government, firm),
        "epsilon$(parameterstring(signal.ϵ))",
        "sigma$(parameterstring(signal.σ))",
    ), "_")
end