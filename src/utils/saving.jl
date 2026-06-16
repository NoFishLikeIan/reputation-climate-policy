import JLD2
import Printf

function parameterstring(x)
    return replace(Printf.@sprintf("%.3e", x), "+" => "")
end

function dynamicsolutionlabel(firm)
    return "omega$(parameterstring(firm.ω))_nu$(parameterstring(firm.ν))"
end

function solutionlabel(climate, government, firm, signal)
    return join((
        "e0$(parameterstring(firm.e₀))",
        "nu$(parameterstring(firm.ν))",
        "y0$(parameterstring(government.y₀))",
        "delta$(parameterstring(government.δ))",
        "r$(parameterstring(government.r))",
        "gamma$(parameterstring(climate.γ))",
        "zeta$(parameterstring(climate.ζ))",
        "epsilon$(parameterstring(signal.σ))",
        "sigma$(parameterstring(signal.ϵ))",
    ), "_")
end