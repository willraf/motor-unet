# MotorUNet
Sim-to-Real Learning for Identifying Motor Unit Action Potentials from Low-Density Surface EMG Recordings


## Environment setup

```bash
pip install -r requirements.txt
```


## Data Preparation

- **Simulated Data:** Upload your simulated MUAP library (`.pkl`, `.mat`, or `.npy`) with shape `(num_muaps, num_channels, time_steps)`. Update `data.muap_pkl_path` in `configs/main.yaml`.
    
- **Experimental Data (Optional):** Upload real sEMG recordings (`.npy` or `.mat`) with shape `(num_channels, time_steps)`. Update the prediction section in `configs/main.yaml`.
    
## Run full train-eval-predict pipeline

```bash
python -m scripts.main --config configs/main.yaml
```

## Or run each part separately

```bash
python -m scripts.train --config configs/main.yaml
```

```bash
python -m scripts.eval --config configs/main.yaml
```

```bash
python -m scripts.predict --config configs/main.yaml
```


## Citation

```
@inproceedings{raftery2026motorunet,
  title={MotorUNet: Sim-to-Real Learning for Identifying Motor Unit Action Potentials from Low-Density Surface EMG Recordings},
  author={Raftery, William and Ezaz-Nikpay, Noura and Maksymenko, Kostiantyn and Grison, Agnese and Mamidanna, Pranav and Farina, Dario},
  booktitle={IEEE Engineering in Medicine and Biology Society (EMBC)},
  year={2026}
}
```

