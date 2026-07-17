import ForwardDiff

@inline rootvalue(x) = x
@inline rootvalue(x::ForwardDiff.Dual) = rootvalue(ForwardDiff.value(x))

"""
    brent(f, (lower, upper); xatol, xrtol, maxiters)

Find a bracketed root using Brent's method. Arithmetic retains dual numbers, while
signs, ordering and convergence are evaluated using their primal values.
"""
function brent(f, bracket::Tuple{A, B}; xatol = nothing, xrtol = nothing, maxiters = 100) where {A <: Real, B <: Real}
    a, b = bracket
    fa, fb = f(a), f(b)
    a, b, fa, fb = promote(a, b, fa, fb)
    pfa, pfb = rootvalue(fa), rootvalue(fb)

    isfinite(pfa) && isfinite(pfb) || throw(ArgumentError("the bracket has non-finite endpoints"))
    iszero(pfa) && return a
    iszero(pfb) && return b
    signbit(pfa) == signbit(pfb) && throw(ArgumentError("the interval does not bracket a root"))

    if abs(pfa) < abs(pfb)
        a, b = b, a
        fa, fb = fb, fa
    end

    T = promote_type(typeof(rootvalue(a)), typeof(rootvalue(b)))
    atol = isnothing(xatol) ? eps(T) : abs(rootvalue(xatol))
    rtol = isnothing(xrtol) ? eps(T) : abs(rootvalue(xrtol))

    c, fc = a, fa
    d = c
    bisected = true

    for _ in 1:maxiters
        pfa, pfb, pfc = rootvalue(fa), rootvalue(fb), rootvalue(fc)

        s = if pfa != pfc && pfb != pfc
            a * fb * fc / ((fa - fb) * (fa - fc)) +
            b * fa * fc / ((fb - fa) * (fb - fc)) +
            c * fa * fb / ((fc - fa) * (fc - fb))
        else
            b - fb * (b - a) / (fb - fa)
        end

        pa, pb, pc, pd = rootvalue(a), rootvalue(b), rootvalue(c), rootvalue(d)
        ps = rootvalue(s)
        lower, upper = minmax((3pa + pb) / 4, pb)
        tolerance = max(atol, rtol * max(abs(pb), abs(pc), abs(pd)))

        rejectstep = !isfinite(ps) ||
            !(lower < ps < upper) ||
            (bisected && abs(ps - pb) >= abs(pb - pc) / 2) ||
            (!bisected && abs(ps - pb) >= abs(pc - pd) / 2) ||
            (bisected && abs(pb - pc) <= tolerance) ||
            (!bisected && abs(pc - pd) <= tolerance)

        if rejectstep
            s = (a + b) / 2
            bisected = true
        else
            bisected = false
        end

        fs = f(s)
        pfs = rootvalue(fs)
        isfinite(pfs) || throw(ArgumentError("the objective is non-finite inside the bracket"))
        iszero(pfs) && return s

        d = c
        c, fc = b, fb

        if signbit(pfa) != signbit(pfs)
            b, fb = s, fs
        else
            a, fa = s, fs
        end

        if abs(rootvalue(fa)) < abs(rootvalue(fb))
            a, b = b, a
            fa, fb = fb, fa
        end

        pa, pb = rootvalue(a), rootvalue(b)
        tolerance = max(atol, rtol * max(abs(pa), abs(pb)))
        abs(pb - pa) <= tolerance && return b
    end

    throw(ErrorException("Brent's method did not converge"))
end
