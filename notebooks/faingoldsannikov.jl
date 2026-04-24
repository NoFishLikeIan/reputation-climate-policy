### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ d843633c-f12e-11f0-8e75-d943d67c24f5
using Plots, LaTeXStrings

# ╔═╡ 8c7ebeb2-fd73-4872-ac0b-190f9eb5462b
using PlutoUI

# ╔═╡ 3ba13a81-ac25-4cd9-851f-26ac21b4fe33
using Optim, FastClosures

# ╔═╡ 84f3a07a-81d0-49f5-b685-a6b51a1cefae
using UnPack

# ╔═╡ 03574177-09a7-4709-9800-8ecb41f24fa5
using Roots

# ╔═╡ 19f87f8f-21c4-49d7-a1b3-7c5a44b243ea
using DifferentialEquations

# ╔═╡ 34e6de51-e986-4562-a3e6-eb03c3a23149
using StaticArrays

# ╔═╡ bb73d5f6-6685-473f-9a69-dd4798645158
using BenchmarkTools

# ╔═╡ bd19032e-02e1-4fe4-aecd-b54a3f0feafc
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 2000px;
    	padding-left: max(160px, 20%);
    	padding-right: max(160px, 20%);
	}
</style>
"""

# ╔═╡ a9de090c-090f-4930-be8d-02c2a107510f
default(dpi = 180, linewidth = 2.5, label = false, background_color = :transparent)

# ╔═╡ 74f7a92d-f820-497f-af41-3aa6f534b55d
TableOfContents()

# ╔═╡ 8fae1b6b-015e-43c6-ad5b-890c8a65305c
md"
# Utils
## Plotting utils
"

# ╔═╡ b58e1d2e-e218-4d58-9c2d-c47cab866560
begin
	τspace = 0:1:250
	τticks = 0:50:maximum(τspace)
	τticklabels = [L"%$x" for x in τticks]
	τlabel =  L"Carbon tax $\tau \; \textrm{USD} / \textrm{tC}$"
	
	A = 0:0.01:1
	aticks = 0:0.2:1
	aticklabels = [L"%$(floor(Int, 100x)) \%" for x in aticks]
	alabel = L"Abatement rate $a$"
end;

# ╔═╡ c18420d2-b525-4dd7-8de6-d60cdf1a3c4b
begin
	function plotvectorfield(xs, ys, g::Function; plotkwargs...)
	    fig = plot()
	    plotvectorfield!(fig, xs, ys, g; plotkwargs...)
	    return fig
	end
	
	function plotvectorfield!(figure, xs, ys, g::Function; rescale = 1, plotkwargs...)
	
		xlims = extrema(xs)
		ylims = extrema(ys)
		
		N, M = length(xs), length(ys)
		xm = repeat(xs, outer=M)
		ym = repeat(ys, inner=N)
		
		field = g.(xm, ym)
		
		scale = rescale * (xlims[2] - xlims[1]) / min(N, M)
		u = @. scale * first(field)
		v = @. scale * last(field)
		
		steadystates = @. (u ≈ 0) * (v ≈ 0)
		
		u[steadystates] .= NaN
		v[steadystates] .= NaN
		
		z = (x -> √(x'x)).(field)
		
	    quiver!(
	        figure, xm, ym;
	        quiver = (u, v), line_z=repeat(z, inner=4),
	        aspect_ratio = 1, xlims = xlims, ylims = ylims,
	        c = :batlow, colorbar = false,
	        plotkwargs...
	    )
	
	end
end

# ╔═╡ efed6f54-eb6b-4f98-9507-c8d601d45e76
md"## Constants"

# ╔═╡ 5602e190-2d3b-4ae2-9ee9-58a9cb3b73c1
begin
	const CtoCO₂ = 44 / 12
	const e₀ = 37.8 / CtoCO₂
	const y₀ = 197.231
	const dicescc = 66 / 1000
	const dietzφ = 3e-5
	const taxfactor = CtoCO₂ * 1e9 * 1e-12; # $ / tCO₂ → t$ / GtC
end;

# ╔═╡ ca2c45d8-393b-415e-baaa-d38306664dca
md"
# Reputation in continuous time

This is an attempt to employ the framework by [Faingold & Sannikov (Econometrica, 2011)](https://doi.org/10.3982/ECTA7377).
"

# ╔═╡ edca9b01-5497-46c4-9879-4a1e5c2f87ec
md"
## Model
### Firms

A continuum of polluting energy firms $i \in [0, 1]$ are endowed with a production technology emitting $e_{0}$. Firms can choose to invest in abatement efforts $a_{i, t}$, at a cost $c(a_{i, t})$, to reduce their emissions 

$\begin{equation}
    e_{i, t} = (1 - a_{i, t}) e_{0}.
\end{equation}$ All firms pay a carbon tax $\tau_t$ on emissions such that their total costs are given by 

$\begin{equation}
    k(a_{i, t}, \tau_t) = \underbrace{(1 - a_{i, t}) e_{0}}_{e_{i, t}} \tau_t + c(a_{i, t}),
\end{equation}$ where $c(a) = \nu \frac{a^2}{2}.$

Aggregate emissions are given by $e_t = \int_0^\infty e_{i, t} \mathrm{d}i = (1 - a_t) e_0$ where $a_t = \int_0^\infty a_{i, t} \mathrm{d}i.$
"

# ╔═╡ a7357976-2f2c-497f-9ab0-bdd9a809df1a
Base.@kwdef struct Firm{T}
	e₀::T = e₀ # emissions [GtC/year]
	ν::T = dietzφ * y₀ * (e₀ * CtoCO₂)^2 # adjustment costs [year / tEur²]
end

# ╔═╡ 05306d65-3c46-421c-90ec-bf2e665d2ca7
function c(a, firm::Firm)
	firm.ν * a^2 / 2
end;

# ╔═╡ 4f10b1c7-0dd1-4128-b224-fc872e4b7276
function e(a, firm::Firm)
	(1 - a) * firm.e₀
end;

# ╔═╡ 49daa43b-65a9-44cf-9204-5c1b9487254c
function k(a, τ, firm::Firm)
	e(a, firm) * τ + c(a, firm)
end;

# ╔═╡ e1cf6b4c-744b-44ec-92a3-d18af362bb68
firm = Firm();

# ╔═╡ 3ba4fb2e-b3ca-49bc-a73e-2a2bccdb3af2
md"
### Government

Aggregate emissions $e_{t}$ generate damages 

$d(e_t) \coloneqq \frac{\xi}{2} e_t^2$

as a fraction of output $y_0$. We assume that $d'(e_0) = \xi e_0 \approx 0.2 \; \mathrm{tUSD} / \mathrm{GtC}$. The government sets carbon taxes $\tau_t$ to minimise the net present value of the green transition's welfare costs 
$\begin{equation}
    \int_{0}^{\infty} r e^{-r t} w(\bar{a}_t, \tau_t) \mathrm{d}t
\end{equation}$ where 

$\begin{equation}
    w(\tau_t, \bar{a}_t) \coloneqq y_0 \; d(\bar{a}_t) + c\left(\bar{a}_t\right) + l(\tau_t)
\end{equation}$ where 

$\begin{equation}
	l(\tau) = \frac{\delta}{2} \tau^2
\end{equation}$ 

is a deadweight loss from carbon taxation.
"

# ╔═╡ f5c5fcd0-c43b-4abf-886f-5a3022affea6
const matchscc = 66 / 1000;

# ╔═╡ 855f60fe-a2be-4baa-8456-132de601b01a
Base.@kwdef struct Gov{T}
	ξ::T = dicescc / e₀ # linear damage coefficient [-]
    y₀::T = y₀ # output/GDP [trillion Eur/year]
	δ::T = 0.2 * y₀
	r::T = 0.05
end

# ╔═╡ 146425a3-008b-4c49-b974-611558563c32
function d(e, gov::Gov, firm::Firm)
	(gov.ξ / 2) * e^2
end;

# ╔═╡ bd4538ef-9fbe-4a51-a3d7-1655b08935b6
function l(τ, gov::Gov)
	(gov.δ / 2) * τ^2
end;

# ╔═╡ d5df8b25-7510-40cd-b0fe-512bb8199390
w(τ, a, gov::Gov, firm::Firm) = gov.y₀ * d(e(a, firm), gov, firm) + c(a, firm) + l(τ, gov);

# ╔═╡ 2455a855-b249-454f-ae9d-3c7794b96b5e
gov = Gov();

# ╔═╡ 145c4934-4ce9-4d7a-bcc6-fba053f78eea
let
	fig = plot(xlabel = alabel, xlims = (0, 1), ylabel = L"Damages $\mathrm{tUSD /year}$", ylims = (0, Inf), xticks = (aticks, aticklabels))
	plot!(A, a -> gov.y₀ * d(e(a, firm), gov, firm); c = :darkgreen, label = L"d(a) y_0")
	plot!(A, a -> c(a, firm); c = :darkred, label = L"c(a)")
	
end

# ╔═╡ dd31ae41-3b6d-41d1-8b8b-95408ff2133f
const τ₀ = 3 * taxfactor;

# ╔═╡ c7dfd8c7-1b38-4fc1-8162-0472e7a090ce
let
	contourf(A, τspace, (a, τ) -> w(τ * taxfactor, a, gov, firm); xticks = (aticks, aticklabels), yticks = (τticks, τticklabels), xflip = false, c = :Reds, linewidth = 0.5, xlabel = alabel, ylabel = τlabel, title = L"Social costs $w(\tau, a) \; \textrm{tUSD} / \textrm{year}$", clims = (0, Inf))
end

# ╔═╡ af101b48-00d7-4fca-9c7e-7365687fc03f
md"
## Committed Policies

Under comittment $\tau_t = \tau^{\mathrm{c}}$, the firm takes $\tau^{\mathrm{c}}$ as given and chooses $a \in [0, 1]$ to minimise $k(a, \tau^c)$. The minimiser is given by $a^{\mathrm{c}}(\tau^\mathrm{c}) = \min\left\{\frac{e_0}{\nu} \tau^{\mathrm{c}}, 1\right\}$.

Then, the problem of a committed government amounts to minimising $w^{\mathrm{c}}(\tau) = w(\tau^{\mathrm{c}}, a^{\mathrm{c}}(\tau^\mathrm{c})))$ which yields

$\begin{equation}
\tau^{\mathrm{c}} = \frac{y_0 \xi e_0 \left(\frac{e_0^2}{\nu}\right)}{y_0 \xi \left(\frac{e_0^2}{\nu}\right)^2 + \left(\frac{e_0^2}{\nu}\right) + \delta}
\end{equation}$

"

# ╔═╡ 03385be6-b623-4e24-b0c0-eca701d37e77
function aᶜ(τ, firm::Firm)
	min((firm.e₀ / firm.ν) * τ, 1)
end;

# ╔═╡ 846a0839-a3e1-4288-af53-5d20915cf73f
begin
	@unpack ξ, δ = gov
	@unpack ν = firm

	mc = e₀^2 / ν
	
	num = ξ * y₀ * mc * e₀
	den = ξ * y₀ * mc^2 + mc + δ

	const τᶜ = max((num / den) / taxfactor, ν / e₀) * taxfactor
	const wᶜ = w(τᶜ, aᶜ(τᶜ, firm), gov, firm)
end;

# ╔═╡ 737e6fad-e711-4a52-bbae-3c9f66830573
let
	fig = plot(τspace, τ -> w(τ * taxfactor, aᶜ(τ * taxfactor, firm), gov, firm), xlims = extrema(τspace), c = :black, xlabel = τlabel, xticks = (τticks, τticklabels), ylabel = L"Social costs $w^{\mathrm{c}}(\tau) \; \textrm{tUSD} / \textrm{year}$", yguidefontcolor = :black)
	
	vline!(fig, [τᶜ / taxfactor]; c = :black, linestyle = :dash)
	
	plot!(twinx(fig), τspace, τ -> aᶜ(τ * taxfactor, firm), xlims = extrema(τspace), ylims = (0, 1.01), c = :darkgreen, ylabel = alabel, yguidefontcolor = :darkgreen)
end

# ╔═╡ 6607f6fa-88a0-473c-a499-310fd935c174
md"## Reputation"

# ╔═╡ 2fc61a7f-745d-42c0-a8f0-4ac2a34bbb6e
md"
### Signal

Firms do not observe $\tau_t$ directly, but a signal $s_t$ following 

$\begin{equation}
	\mathrm{d}s_t = \mu(\tau_t, \bar{a}_t) \mathrm{d} t + \sigma\mathrm{d}Z_t.
\end{equation}$

We assume that the signal $s_t$ is growing in the policy rate $\tau$ and decreasing in the abatement rate $a_t$, that is $\mu(\tau, a) \coloneqq \alpha \tau - a$.

The firms form a belief $\phi_t$ on the probability of the government being committed, with prior $\phi_0$.
"

# ╔═╡ dc05d560-9f17-4844-9609-c1105fbec573
Base.@kwdef struct Signal{T <: Real}
	σ::T = √τ₀
	α::T = τ₀ * 0.05
	ϵ::T = 1e-2
end

# ╔═╡ 2a4c171d-07aa-4459-bf2e-27ebb2a97bab
function μ(τ, a, signal::Signal)
	signal.ϵ * (τ - signal.α * a)
end;

# ╔═╡ 8c71d228-5fe6-45ce-b108-c6b5853989a8
signal = Signal();

# ╔═╡ 17cf162b-9a35-4301-a465-fbf4ba93c49c
md"
### Government

Introduce the reputational weight $z \in \mathbb{R}$. The optimal tax policy $\tau$, given an abatement level $\bar{a}$ and the reputational weight $z$ is given by 

$\begin{equation}
	\begin{split}
		\tau \in \arg_{\tau}\min \; &\mathcal{L}(\tau; \bar{a}, z) \text{ where } \\
		&\mathcal{L}(\tau; \bar{a}, z) = w(\tau, \bar{a}) - z \frac{ \mu(\tau^{\mathrm{c}}, \bar{a}) - \mu(\tau, \bar{a})}{\sigma^2} \mu(\tau, \bar{a}).
	\end{split}
\end{equation}$ 

First order condition yields the optimal policy 

$\begin{equation}
	\tau_t = \frac{\tau^c + \alpha \bar{a}_t}{2 +\frac{\delta \sigma^2}{\varepsilon^2 z}}.
\end{equation}$
"

# ╔═╡ ae44f829-e754-4879-8ff5-2cde5c8c7f28
function L(τ, a, z, signal::Signal, gov::Gov, firm::Firm)
	w(τ, a, gov, firm) - z * μ(τ, a, signal) * (μ(τᶜ, a, signal) - μ(τ, a, signal)) / signal.σ^2
end;

# ╔═╡ 29a3bcd6-7348-4398-8fbd-267f9212c119
function reputationweight(z, signal::Signal, gov::Gov)
	@unpack ϵ, σ = signal
	@unpack δ = gov
	
	return (δ * σ^2) / (ϵ^2 * z)
end;

# ╔═╡ cf272c45-774b-4384-8b65-df58bfde5eb3
function optimalτ(a, z, signal::Signal, gov::Gov, firm::Firm)
	@unpack α, ϵ, σ = signal
	@unpack δ = gov

	(τᶜ + α * a) / (2 + reputationweight(z, signal, gov))
end;

# ╔═╡ 483185a7-7ca6-470e-8d67-1e6ef304d19c
begin
	z̲ = 0.
	z̄ = 10_000
end;

# ╔═╡ 16c84394-1525-409a-bb58-215756926056
md"
- ``a =`` $(@bind La Slider(A, show_value = true, default = 0.5))
- ``z =`` $(@bind Lz Slider(range(z̲, z̄, 201), show_value = true, default = 0.))
"

# ╔═╡ 860109da-f4c4-4247-8fe3-f6f7c1b8e331
let
	cmin = :black
	cmax = beliefscolors[:green]

	zmin = 0.
	zmax = 10_000
	
	fig = plot(xlabel = τlabel, xticks = (τticks, τticklabels), legendtitle = L"Reputation $z$", legendtitlefontsize = 9, legendfontsize = 9, ylabel = L"Welfare $\mathcal{L}(\tau; %$(La), z) \textrm{tUSD} / \textrm{year}$", legend = :topleft)
	
	plot!(fig, τspace, τ -> L(τ * taxfactor, La, zmin, signal, gov, firm), label = L"%$zmin", c = cmin)

	cweight = (Lz - zmin) / (zmax - zmin)
	c = get(cgrad([cmin, cmax]), cweight)

	minimizer = optimalτ(La, Lz, signal, gov, firm)
	minimum = L(minimizer, La, Lz, signal, gov, firm)
			
	plot!(fig, τspace, τ -> L(τ * taxfactor, La, Lz, signal, gov, firm), label = L"%$Lz", c = c)
	scatter!(fig, [minimizer / taxfactor], [minimum], c = :black)
	annotate!(fig, minimizer / taxfactor, minimum, text(L"\tau = %$(round(minimizer / taxfactor, digits = 2))", 10, :bottom))
	
	plot!(fig, τspace, τ -> L(τ * taxfactor, La, zmax, signal, gov, firm), label = L"%$zmax", c = cmax)


	fig
end

# ╔═╡ fdb1300b-ba5d-4337-bb52-fc7b23b4fb13
let
	zspace = range(z̲, z̄; length = 101)
	τ̄ = optimalτ(La, Inf, signal, gov, firm)
	
	fig = plot(xlabel = L"Reputation weight $z$", ylabel = τlabel, ylims = (0, 1.01τ̄ / taxfactor), xlims = (0, Inf))
	plot!(fig, zspace, z -> optimalτ(La, z, signal, gov, firm) / taxfactor; c = :black)

	hline!(fig, [τ̄ / taxfactor]; c = :black, linestyle = :dash)

	fig
end

# ╔═╡ be015c5a-e9ab-45dc-a9ab-171a6322034e
md"
### Firm

The firms, given the subjective probability $\phi$ on the government being committed, choose

$\begin{equation}
	\begin{split}
		\bar{a} \in \arg_\bar{a}\min \; &k^{\phi}(\bar{a}; \tau, \phi) \text{ where } \\
		&k^{\phi}(\bar{a}; \tau, \phi) = \phi k(\bar{a}, \tau^{\mathrm{c}}) + (1 - \phi) k(\bar{a}, \tau).
	\end{split}
\end{equation}$ 

Assuming symmetry among firms, the optimal policy is given by 

$\begin{equation}
	a_t = \frac{e_0}{\nu} \left( \phi \tau^c + (1 - \phi) \tau_t \right).
\end{equation}$
"

# ╔═╡ 47e0c197-7bdd-4bda-b4cd-06c7aeec639c
function K(a, τ, ϕ, firm::Firm)
	ϕ * k(a, τᶜ, firm) + (1 - ϕ) * k(a, τ, firm)
end;

# ╔═╡ 022afc7d-1250-40a3-9848-f1d332cf1dc2
md"
- ``\phi =`` $(@bind Kϕ Slider(0:0.01:1, show_value = true, default = 0.5))
- ``\tau =`` $(@bind Kτ Slider(τspace, show_value = true, default = τ₀ / taxfactor)) ``\textrm{USD} / \textrm{tC}``
"

# ╔═╡ a9a75f0c-e59e-40a7-ade5-cef40353071d
function aᵒ(τ, ϕ, firm::Firm)
	min(firm.e₀ * (ϕ * τᶜ + (1 - ϕ) * τ) / firm.ν, 1)
end;

# ╔═╡ 19acba30-d3d7-49a2-8c96-2df68b232196
let
	cmin = :darkred
	cmax = :darkgreen

	fig = plot(xlabel = alabel, xticks = (aticks, aticklabels), ylims = (0, Inf))
	plot!(fig, A, a -> K(a, Kτ * taxfactor, 0., firm); label = L"0 \%", c = cmin, ylabel = L"Costs $k^{\phi}(\tau; %$(La), z) \textrm{tUSD} / \textrm{year}$")
	
	c = get(cgrad([cmin, cmax]), Kϕ)
	plot!(fig, A,  a -> K(a, Kτ * taxfactor, Kϕ, firm); label = L"%$(floor(Int, 100Kϕ)) \%", c = c)

	minimizer = aᵒ(Kτ * taxfactor, Kϕ, firm)
	minimum = K(minimizer, Kτ * taxfactor, Kϕ, firm)
	scatter!(fig, [minimizer], [minimum], c = :black)
	annotate!(fig, minimizer, minimum, text(L"a = %$(round(Int, 100minimizer)) \%", 10, :bottom))
	
	plot!(fig, A, a -> K(a, Kτ * taxfactor, 1., firm); label = L"100 \%", c = cmax)
	
end

# ╔═╡ 2688fc63-164f-4fd0-bc8f-abbde845d51d
md"
The equilibrium tax gives 

$\begin{equation}
	\tau_t = \tau^*(\phi_t, z_t) \coloneqq \frac{1 + \alpha \frac{e_0}{\nu} \phi}{2 + \frac{\delta \sigma^2}{\varepsilon^2 z} - \alpha \frac{e_0}{\nu} (1 - \phi)} \tau^c.
\end{equation}$
"

# ╔═╡ 45e18af0-2531-4f70-a30d-a65f2f3560b6
function τᵒ(ϕ, z, signal::Signal, gov::Gov, firm::Firm)
	@unpack α, σ = signal
	@unpack ν, e₀ = firm
	@unpack δ = gov

	weight = reputationweight(z, signal, gov)

	num = 1 + α * (e₀ / ν) * ϕ
	den = 2 + weight - α * (e₀ / ν) * (1 - ϕ)

	return (num / den) * τᶜ
end;

# ╔═╡ a1513476-8abf-40f6-bd39-d8d424424280
let
	ϕs = range(0, 1, 101)
	zspace = range(z̲, z̄, 101)
	
	surface(ϕs, zspace, (ϕ, z) -> τᵒ(ϕ, z, signal, gov, firm) / τᶜ; xlabel = L"Reputation $\phi$", ylabel = L"Reputation weight $z$", title = L"Tax ratio $\tau / \tau^{\mathrm{c}}$", c = :Reds, linewidth = 0., clims = (0, 1), zlims = (0, 1))
end

# ╔═╡ a0dea2de-084a-4c9b-95b9-e789d03b6c52
md"## Value function"

# ╔═╡ 9cb3a94e-b549-4add-bb92-5c58037a8e54
md"
The net present welfare is defined as 

$\begin{equation}
	W_t = \mathbb{E}_t \int_t^\infty \rho e^{-\rho (s - t)} w(\tau_t, \bar{a}_t) \; \mathrm{d}s
\end{equation}$

Given the two equilibrium policy functions $a(\phi, z)$ and $\tau(\phi, z)$, and assuming that $W_t = u(\phi_t)$, then $u$ must satisfy the second order differential equation

$\begin{equation}
	u''(\phi) = \frac{2u'(\phi)}{1 - \phi} + 2r \left(\frac{\sigma}{\alpha \phi (1 - \phi)} \right)^2 \frac{u(\phi) - w^*(\phi, z_t)}{(\tau^c - \tau^*(\phi, z_t))^2}
\end{equation}$

where 

$\begin{equation}
	w^*(\phi, z) = w(\tau(\phi, z), a(\phi, z)).
\end{equation}$

Using the definition of $z$, this can be turned into a two-dimensional first order ODE as follows

$\begin{equation}
    \begin{cases}
        u' = r \frac{z}{\phi (1 - \phi)} \\
        z' = \frac{1}{\phi (1 - \phi)} \left(z + 2\left(\frac{\sigma}{\alpha (\tau - \tau^*(\phi, z))}\right)^2 (u - w^*(\phi, z)) \right)
    \end{cases}
\end{equation}$ 

with boundary conditions

$\begin{align}
    u(0) &= w(0, 0) \\
    u(1) &= w(\tau^{\mathrm{c}}, a^{\mathrm{c}}) \\
    z(0) &= z(1) = 0.
\end{align}$

"

# ╔═╡ bb77dda6-8373-4e79-a420-51e5058fe7ec
function wᵒ(ϕ, z, signal::Signal, gov::Gov, firm::Firm)
	τ = τᵒ(ϕ, z, signal, gov, firm)
	a = aᵒ(τ, ϕ, firm)

	return w(τ, a, gov, firm)
end;

# ╔═╡ 07f780e7-7f41-403a-bd59-21465a45f83a
sigmoid(x) = inv(1 + exp(-x));

# ╔═╡ 3e036bce-9b36-4dd3-8a31-b02f629ddfd4
function F!(dx, x, p, ψ)
	signal, gov, firm = p
	u, z = x

	@unpack α, σ = signal
	@unpack r = gov

	ϕ = sigmoid(ψ)
	
	τ = τᵒ(ϕ, z, signal, gov, firm)
	w = wᵒ(ϕ, z, signal, gov, firm)

	dμ = α * (τᶜ - τ)
	
	dx[1] = r * z
	dx[2] = (z + 2 * (σ^2 / dμ)^2 * (u - w))
end;

# ╔═╡ 286acc96-e38e-434e-b9b5-2cf2f7a19ca2
function leftbc!(res, x₀, p)
    signal, gov, firm = p
    u₀, z₀ = x₀
	
    res[1] = u₀ - w(0., 0., gov, firm)
	res[2] = z₀
end;

# ╔═╡ b55e85b4-e342-4140-9f6a-2a53e5bfbb48
function rightbc!(res, x₁, p)
    signal, gov, firm = p
    u₁, z₁ = x₁
    
    res[1] = u₁ - w(τᶜ, aᶜ(τᶜ, firm), gov, firm)
	res[2] = z₁
end;

# ╔═╡ e9e7fab3-57f4-4893-84e5-147eacb00aaf
begin
	w̄ =  (w(0., 0., gov, firm) + w(τᶜ, aᶜ(τᶜ, firm), gov, firm)) / 2
	x₀ = [w̄, 0.1]
	M = 10.
	ϕspan = (-M, M)
	p = (signal, gov, firm)
	bcresid_prototype = (zeros(2), zeros(2))
	
	bvp = TwoPointBVProblem(F!, (leftbc!, rightbc!), x₀, ϕspan, (signal, gov, firm); bcresid_prototype)
end

# ╔═╡ 6fc2368d-a580-4afa-95e1-ec570f63a41f
let
	dx = similar(x₀)
	F!(dx, x₀, p, 0.1)

	@code_warntype F!(dx, x₀, p, 0.1)
end

# ╔═╡ f99168ac-2fce-4179-a3dc-0278024fe150
sol = solve(bvp,  MIRK4(), dt = 0.001)

# ╔═╡ a8614217-e1de-4ee7-9936-61269e0cce42
plot(sol, idxs = 1)

# ╔═╡ c651d581-69c9-447c-8434-9a33a4f8ff94
plot(sol, idxs = 2)

# ╔═╡ 53e4d7a6-7976-4d90-9b09-4c12835b501d
plot(0:0.01:1, ϕ -> sol(log(ϕ / (1 - ϕ)))[1])

# ╔═╡ Cell order:
# ╟─bd19032e-02e1-4fe4-aecd-b54a3f0feafc
# ╠═d843633c-f12e-11f0-8e75-d943d67c24f5
# ╠═a9de090c-090f-4930-be8d-02c2a107510f
# ╠═8c7ebeb2-fd73-4872-ac0b-190f9eb5462b
# ╠═74f7a92d-f820-497f-af41-3aa6f534b55d
# ╠═3ba13a81-ac25-4cd9-851f-26ac21b4fe33
# ╠═84f3a07a-81d0-49f5-b685-a6b51a1cefae
# ╠═03574177-09a7-4709-9800-8ecb41f24fa5
# ╟─8fae1b6b-015e-43c6-ad5b-890c8a65305c
# ╠═b58e1d2e-e218-4d58-9c2d-c47cab866560
# ╟─c18420d2-b525-4dd7-8de6-d60cdf1a3c4b
# ╟─efed6f54-eb6b-4f98-9507-c8d601d45e76
# ╠═5602e190-2d3b-4ae2-9ee9-58a9cb3b73c1
# ╟─ca2c45d8-393b-415e-baaa-d38306664dca
# ╟─edca9b01-5497-46c4-9879-4a1e5c2f87ec
# ╠═a7357976-2f2c-497f-9ab0-bdd9a809df1a
# ╠═05306d65-3c46-421c-90ec-bf2e665d2ca7
# ╠═4f10b1c7-0dd1-4128-b224-fc872e4b7276
# ╠═49daa43b-65a9-44cf-9204-5c1b9487254c
# ╠═e1cf6b4c-744b-44ec-92a3-d18af362bb68
# ╟─3ba4fb2e-b3ca-49bc-a73e-2a2bccdb3af2
# ╠═f5c5fcd0-c43b-4abf-886f-5a3022affea6
# ╠═855f60fe-a2be-4baa-8456-132de601b01a
# ╠═146425a3-008b-4c49-b974-611558563c32
# ╠═bd4538ef-9fbe-4a51-a3d7-1655b08935b6
# ╠═d5df8b25-7510-40cd-b0fe-512bb8199390
# ╠═2455a855-b249-454f-ae9d-3c7794b96b5e
# ╟─145c4934-4ce9-4d7a-bcc6-fba053f78eea
# ╠═dd31ae41-3b6d-41d1-8b8b-95408ff2133f
# ╟─c7dfd8c7-1b38-4fc1-8162-0472e7a090ce
# ╟─af101b48-00d7-4fca-9c7e-7365687fc03f
# ╠═03385be6-b623-4e24-b0c0-eca701d37e77
# ╠═846a0839-a3e1-4288-af53-5d20915cf73f
# ╟─737e6fad-e711-4a52-bbae-3c9f66830573
# ╟─6607f6fa-88a0-473c-a499-310fd935c174
# ╟─2fc61a7f-745d-42c0-a8f0-4ac2a34bbb6e
# ╠═dc05d560-9f17-4844-9609-c1105fbec573
# ╠═2a4c171d-07aa-4459-bf2e-27ebb2a97bab
# ╠═8c71d228-5fe6-45ce-b108-c6b5853989a8
# ╟─17cf162b-9a35-4301-a465-fbf4ba93c49c
# ╠═ae44f829-e754-4879-8ff5-2cde5c8c7f28
# ╠═29a3bcd6-7348-4398-8fbd-267f9212c119
# ╠═cf272c45-774b-4384-8b65-df58bfde5eb3
# ╠═483185a7-7ca6-470e-8d67-1e6ef304d19c
# ╟─16c84394-1525-409a-bb58-215756926056
# ╟─860109da-f4c4-4247-8fe3-f6f7c1b8e331
# ╟─fdb1300b-ba5d-4337-bb52-fc7b23b4fb13
# ╟─be015c5a-e9ab-45dc-a9ab-171a6322034e
# ╠═47e0c197-7bdd-4bda-b4cd-06c7aeec639c
# ╟─022afc7d-1250-40a3-9848-f1d332cf1dc2
# ╠═a9a75f0c-e59e-40a7-ade5-cef40353071d
# ╟─19acba30-d3d7-49a2-8c96-2df68b232196
# ╟─2688fc63-164f-4fd0-bc8f-abbde845d51d
# ╠═45e18af0-2531-4f70-a30d-a65f2f3560b6
# ╟─a1513476-8abf-40f6-bd39-d8d424424280
# ╟─a0dea2de-084a-4c9b-95b9-e789d03b6c52
# ╠═19f87f8f-21c4-49d7-a1b3-7c5a44b243ea
# ╠═34e6de51-e986-4562-a3e6-eb03c3a23149
# ╠═bb73d5f6-6685-473f-9a69-dd4798645158
# ╟─9cb3a94e-b549-4add-bb92-5c58037a8e54
# ╠═bb77dda6-8373-4e79-a420-51e5058fe7ec
# ╠═07f780e7-7f41-403a-bd59-21465a45f83a
# ╠═3e036bce-9b36-4dd3-8a31-b02f629ddfd4
# ╠═286acc96-e38e-434e-b9b5-2cf2f7a19ca2
# ╠═b55e85b4-e342-4140-9f6a-2a53e5bfbb48
# ╠═e9e7fab3-57f4-4893-84e5-147eacb00aaf
# ╠═6fc2368d-a580-4afa-95e1-ec570f63a41f
# ╠═f99168ac-2fce-4179-a3dc-0278024fe150
# ╠═a8614217-e1de-4ee7-9936-61269e0cce42
# ╠═c651d581-69c9-447c-8434-9a33a4f8ff94
# ╠═53e4d7a6-7976-4d90-9b09-4c12835b501d
