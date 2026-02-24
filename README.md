# Hybrid Microgrid Energy Management System (EMS) System

This project implements a hybrid forecasting + optimization Energy Management System (EMS) for a microgrid as atest-case.

## Overview

The pipeline includes:

1. Synthetic microgrid data generation
2. LSTM-based load and PV forecasting
3. MILP-based battery scheduling using PuLP
4. End-to-end EMS controller pipeline
5. Unit testing for reproducibility

---

## System Architecture

Load / PV Data  
→ Time Series Dataset  
→ LSTM Forecasting  
→ Battery MILP Optimization  
→ Optimal Scheduling

---

## Components Organization

### 1. Synthetic Data
- Daily sinusoidal load profile
- Daytime PV generation
- Time-of-use pricing

### 2. Forecasting
- LSTM model (TensorFlow / Keras)
- Sliding window time series dataset
- MinMax scaling

### 3. Optimization
- MILP formulation
- Battery SOC constraints
- Charging/discharging exclusivity
- Grid import minimization

---

## 4. How to Run

Install dependencies:

```bash
pip install numpy tensorflow scikit-learn pulp matplotlib
```

Run the EMS:

```bash
python optim_EMS.py
```

Run unit tests:

```bash
pytest
```

---

## Unit Testing Procedure

Unit tests for verification of:

- Data generation correctness
- Dataset windowing
- MILP feasibility
- End-to-end pipeline execution

---

## Project Structure

```
working_folder/
│
├── optim_EMS.py
├── test_optim_ems.py
├── README.md
```

---

## Research Extensions

Future upgrades may include but not limited to:

- Physics-guided forecasting
- DRL-based EMS
- Multi-microgrid coordination
- Uncertainty-aware optimization
- Real dataset integration

---

## 📜 License

For research and academic use.
