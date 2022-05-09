'''
Extremely hacky way to benchmark julia code vs number of threads and vs VAR_order

'''
import os

VAR_orders = [1, 10, 20, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]
nthreads = range(1, 65)

StochasticSynapses_jl = './src/ArrayOfStructs.jl'

def get_VAR_order():
    with open(StochasticSynapses_jl, 'r') as f:
        lines = f.readlines()
    pref = 'const VAR_order = '
    for i in range(len(lines)):
        if lines[i].startswith(pref):
            return int(lines[i].replace(pref, '').strip())

orig_order = get_VAR_order()

def change_VAR_order(order):
    # terrifying function that modifies the source code of julia program
    with open(StochasticSynapses_jl, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    pref = 'const VAR_order = '
    for i in range(len(lines)):
        if lines[i].startswith(pref):
            lines[i] = f'{pref}{order}\n'
    with open(StochasticSynapses_jl, 'w', encoding='utf-8') as f:
        f.writelines(lines)

for VAR_order in VAR_orders:
    # change .jl file
    change_VAR_order(VAR_order)
    for nthread in nthreads:
        # start julia with nthread
        # and have it run benchmark script which writes results to a file
        print(f'Running benchmark for VAR_order={VAR_order}, nthreads={nthread}')
        os.system(f'c:/users/tyler/appdata/local/programs/julia-1.7.0/bin/julia.exe --threads {nthread} run_benchmark.jl')



change_VAR_order(orig_order)
