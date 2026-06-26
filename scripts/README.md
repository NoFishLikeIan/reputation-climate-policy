The correct running sequence is.

```bash
julia --project ".\scripts\committed.jl";
julia --project ".\scripts\boundaries.jl";
julia --project ".\scripts\interior.jl"
```

Parameter overrides can be passed on the command line while keeping defaults for omitted values.

```bash
julia --project ".\scripts\committed.jl" --sigma=0.76;
julia --project ".\scripts\boundaries.jl" --sigma=0.76;
julia --project ".\scripts\interior.jl" --sigma=0.76
```
