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
        "omega$(parameterstring(firm.ω))",
        "y0$(parameterstring(government.y₀))",
        "lambda$(parameterstring(λ(government, firm)))",
        "r$(parameterstring(government.r))",
        "gamma$(parameterstring(climate.γ))",
        "zeta$(parameterstring(climate.ζ))",
        "epsilon$(parameterstring(signal.ϵ))",
        "sigma$(parameterstring(signal.σ))",
    ), "_")
end
