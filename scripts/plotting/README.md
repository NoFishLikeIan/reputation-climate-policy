The main scripts for plotting are `prelinaries.jl`, `dynamics.jl`, and `volatility.jl`.

To run a sequence of scripts in powershell use

```powershell
foreach ($sigma in 0.19, 0.38, 0.76, 1.52) {
    julia --project=. .\scripts\plotting\dynamics.jl --sigma=$sigma
}
```