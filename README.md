# StochasticSynapses.jl

This is a Julia implementation of the stochastic synapse model described in this [paper](https://linktopaper)

## Installation

```julia
  import Pkg; Pkg.add(url="https://github.com/thennen/StochasticSynapses.jl")
```

## Examples

#### CPU version

```julia
using StochasticSynapses
M = 2^20
cells = [Cell() for m in 1:M]
voltages = fill(-1f0, M)
applyVoltage!.(cells, voltages)
I = Ireadout.(cells)
```

#### GPU version

```julia
using StochasticSynapses, CUDA
M = 2^20
p = 10
cells = CellArrayGPU(M, p)
voltages = CUDA.fill(-1f0, M)
applyVoltage!(cells, voltages)
I = Ireadout(cells)
```
