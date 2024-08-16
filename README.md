[![DOI](https://zenodo.org/badge/438966323.svg)](https://zenodo.org/badge/latestdoi/438966323)

# StochasticSynapses.jl

This is a Julia implementation of the stochastic synapse model described in [this paper](https://journal.frontiersin.org/article/10.3389/fnins.2022.941753/full).

See a short demo of the switching operation [here](https://www.youtube.com/watch?v=Kk3HzDUP1Vg).

This model has been superseded by [Synaptogen](https://github.com/thennen/Synaptogen).

## Installation

```julia
  import Pkg; Pkg.add(url="https://github.com/thennen/StochasticSynapses.jl")
```

## Examples

#### CPU version

```julia
using StochasticSynapses
M = 2^20
# Initialize a million cells
cells = [Cell() for m in 1:M]
# SET all cells to their low resistance state
applyVoltage!.(cells, -2f0)
# Apply random voltages to all cells
voltages = randn(Float32, M)
applyVoltage!.(cells, voltages)
I = Iread.(cells)
```

#### GPU version

```julia
using StochasticSynapses, CUDA
M = 2^20
p = 10
cells = CellArrayGPU(M, p)
voltages = CUDA.randn(M)
applyVoltage!(cells, voltages)
I = Iread(cells)
```
