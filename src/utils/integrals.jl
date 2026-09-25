function J(a, b, c)
    x = c / b^2
    if x < 1e-4 # Expand for small c
        return (exp(-a) / b) * (1 - 2x + 12x^2 - 120x^3)
    end

    return exp(-a) * (√π / 2√c) * SF.erfcx(b / 2√c)
end

function G(a, b, c)
    x = c / b^2
    if x < 1e-4
        return (exp(-a) / b^2) * (1 - 6x + 60x^2 - 840x^3)
    end

    return (exp(-a) - b * J(a, b, c)) / 2c
end

function L(a, b, c)
    x = c / b^2

    if x < 1e-4
        return (exp(-a) / b^3) * (2 - 24x + 360x^2 - 6720x^3)
    end

    return ((2c + b^2) * J(a, b, c) - b * exp(-a)) / 4c^2
end