# Kalman Smoothing of non-time-invariant Signals with Partial Information – Combining with the Local Model Fit

## Evaluations
* KS (unknown fixed $R$)
```
python3 main_const_unkR.py
```

* KS+moving average  (unknown sinusoidal $R_t$)
```
python3 main_sin_unkR.py
```

* KS+local fitting  (unknown sinusoidal $R_t$)
```
python3 main_sin_unkR_LF.py
```

* KS (unknown fixed $Q$)
```
python3 main_const_unkQ.py
```

* KS+moving average  (unknown sinusoidal $Q_t$)
```
python3 main_sin_unkQ.py
```

* KS+local fitting  (unknown sinusoidal $Q_t$)
```
python3 main_sin_unkQ_LF.py
```

## Update plots and tables
You need to update plots and tables after you rerun the simulations.
```
python3 export_fig.py
python3 export_table.py
```

## Paralell Computing
The code is paralellized to boost the performance. No set-up is needed when running on your own PC. However, some configurations are needed when running on the ETH Euler cluster. On ETH Euler cluster, follow the instructions below.
1. Move to new software stack. `env2lmod`
2. Load python 3.8 environment. `module load gcc/8.2.0 python/3.8.5`
3. Submit the job `bsub -W <HH>:<MM> -n <Core Count> -R "rusage[mem=<Memory in MB>]" <Instructions>`








