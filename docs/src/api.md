# [API reference](@id api)

Here are the public types and functions, with their arguments and conventions.
For a complete device setup, start with the
[README quickstart](https://github.com/jackhfarrell/FermiSea.jl#quickstart) or the
[square-bells tutorial](@ref square-bells).

```@docs
FermiSea
```

## Fermi surface and transport

```@autodocs
Modules = [FermiSea]
Pages = ["surface.jl", "equations.jl"]
Private = false
```

## Collisions and fields

```@autodocs
Modules = [FermiSea]
Pages = ["collisions.jl", "local_operators.jl"]
Private = false
```

## Boundaries and channels

```@autodocs
Modules = [FermiSea]
Pages = ["boundaries.jl", "channel.jl"]
Private = false
```

## Solving

```@autodocs
Modules = [FermiSea]
Pages = ["steady.jl", "transport_sweep.jl", "moment_sweep.jl"]
Private = false
```

## Observables and storage

```@autodocs
Modules = [FermiSea]
Pages = ["moments.jl", "plot_data.jl", "sampling.jl", "io.jl"]
Private = false
```
