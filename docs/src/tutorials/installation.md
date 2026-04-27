# 1: Installation
To the extent there is anyone interested in this code at all, I expect they will be relatively new to Julia.  This first tutorial just gives my reccomendations to how to get going with Julia and install this package. 

## Install Julia

I recommend using a tool called [`juliaup`](https://docs.julialang.org/en/v1/manual/installation/) to get started. It helps install Julia and 
keeps track of installed Julia versions and makes the `julia` command
available from a terminal.

On macOS or Linux, open a terminal and run:

```bash
curl -fsSL https://install.julialang.org | sh
```

On Windows, open PowerShell and run:

```powershell
winget install julia -s msstore
```

These commands install Juliaup. After the installer finishes, restart your terminal so that the `julia` and `juliaup` commands are available.

## Select Julia 1.11.6

Julia 1.11.6 is recommended for working with this package. Technically the package should support any version after 1.10, but I have tested 1.11.6 mostly since it is the most recent available on the high-performance-computing cluster I have used. 

Install and select Julia 1.11.6 with:

```bash
juliaup add 1.11.6
juliaup default 1.11.6
```

Then check your Julia version from the command line:

```bash
julia --version
```

## Install FermiSea

Start Julia:

```bash
julia
```

Then add FermiSea:

```julia
import Pkg
Pkg.add("FermiSea")
```

To make startup a little faster when you actually try and run a script, you can also run

```julia
Pkg.precompile()
```

Some of the tutorial and interactive scripts use extra packages for plotting,
time stepping, mesh generation, and Literate.jl output. Add those as needed:

```julia
Pkg.add(["CairoMakie", "Gmsh", "Interpolations", "Literate", "OrdinaryDiffEqSSPRK"])
```

## Verify the installation

From the same Julia session, load the package:

```julia
using FermiSea
```

To run the test suite from the command line:

```bash
julia -e 'import Pkg; Pkg.test("FermiSea")'
```
