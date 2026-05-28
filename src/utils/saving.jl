function parameterstring(x)
    return replace(@sprintf("%.3e", x), "+" => "")
end

function dynamicsolutionlabel(firm)
    return "omega$(parameterstring(firm.ω))_nu$(parameterstring(firm.ν))"
end