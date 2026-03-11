# AI-based stability assessment of smart grid

Final Year Project – University of Liverpool  
Author: Anqi Xu  
Department of Electrical Engineering and Electronics  

## Project Overview

Power system stability is a fundamental problem in electrical engineering. Ensuring that the system remains stable after disturbances is essential for a reliable electricity supply.

Lyapunov functions provide a mathematical tool for analysing the stability of nonlinear dynamical systems. However, deriving Lyapunov functions analytically for complex power systems is often difficult and requires significant expert knowledge.

This project investigates **a machine learning approach to automatically generate Lyapunov functions for the power system**.

The project focuses on:

- Constructing symbolic datasets from power system stability models
- Converting mathematical expressions into machine-learning compatible representations
- Training machine learning models to learn Lyapunov functions
- Evaluating the generated Lyapunov candidates for stability analysis

## Stability Model

For a nonlinear dynamical system

dx/dt = f(x)

a Lyapunov function V(x) must satisfy the following conditions:

V(x) > 0  
dV(x)/dt < 0  

If these conditions hold, the system's equilibrium point is stable.

## Project Workflow

The overall workflow of the project is shown below:

Power System Models  
↓  
Extract System Equations  
↓  
Identify Lyapunov Functions  
↓  
Construct Symbolic Dataset  
↓  
Tokenisation of Mathematical Expressions  
↓  
Machine Learning Training  

Finally, this pipeline converts mathematical stability analysis into a machine learning problem.

## Dataset Taxonomy

The dataset in this project is constructed from multiple power-system-related nonlinear dynamical systems and their corresponding Lyapunov functions. Based on the uploaded Python implementations, the dataset can be organised from three perspectives: source, mathematical structure, and state dimension.

## Dataset Summary Table

| ID | System Description | Dimension | Source |
|----|--------------------|-----------|--------|
| 01 | Nonlinear multi-machine swing-equation system | 6D | Trigonometric coupling between rotor angles and frequencies | Quadratic energy-type Lyapunov function | T. L. Vu and K. Turitsyn, "Lyapunov Functions Family Approach to Transient Stability Assessment," in IEEE Transactions on Power Systems, vol. 31, no. 2, pp. 1269-1277, March 2016, doi: 10.1109/TPWRS.2015.2425885. |
| 02 | Reduced two-state nonlinear swing system | 2D | Angle–frequency nonlinear dynamics with sin coupling | Quadratic Lyapunov function with trigonometric terms | T. L. Vu and K. Turitsyn, "Lyapunov Functions Family Approach to Transient Stability Assessment," in IEEE Transactions on Power Systems, vol. 31, no. 2, pp. 1269-1277, March 2016, doi: 10.1109/TPWRS.2015.2425885. |
| 03 | 3-machine Classical power system (non-polynomial swing equations without transfer conductances) | 4D | M. Anghel, F. Milano and A. Papachristodoulou, "Algorithmic Construction of Lyapunov Functions for Power System Stability Analysis," in IEEE Transactions on Circuits and Systems I: Regular Papers, vol. 60, no. 9, pp. 2533-2546, Sept. 2013, doi: 10.1109/TCSI.2013.2246233. |
| 04 | 2-machine vs infinite-bus Classical power system (non-polynomial swing equations with conductances) | 4D | M. Anghel, F. Milano and A. Papachristodoulou, "Algorithmic Construction of Lyapunov Functions for Power System Stability Analysis," in IEEE Transactions on Circuits and Systems I: Regular Papers, vol. 60, no. 9, pp. 2533-2546, Sept. 2013, doi: 10.1109/TCSI.2013.2246233. |
| 05 | Structure-preserving multimachine power system in Lur’e form | 2D | Z. Qiu, C. Duan, W. Yao, P. Zeng and L. Jiang, "Adaptive Lyapunov Function Method for Power System Transient Stability Analysis," in IEEE Transactions on Power Systems, vol. 38, no. 4, pp. 3331-3344, July 2023, doi: 10.1109/TPWRS.2022.3199448.|
| 06 | Polynomial swing-system model with a rational Lyapunov function obtained via optimization | 4D | Rational Lyapunov function | D. Han, A. El-Guindy and M. Althoff, "Power systems transient stability analysis via optimal rational Lyapunov functions," 2016 IEEE Power and Energy Society General Meeting (PESGM), Boston, MA, USA, 2016, pp. 1-5, doi: 10.1109/PESGM.2016.7741322. |
| 07 | Constructed Lyapunov function for a reduced second-order VSG model (non-energy, parameter-dependent) | 2D | Z. Shuai, C. Shen, X. Liu, Z. Li and Z. J. Shen, "Transient Angle Stability of Virtual Synchronous Generators Using Lyapunov’s Direct Method," in IEEE Transactions on Smart Grid, vol. 10, no. 4, pp. 4648-4661, July 2019, doi: 10.1109/TSG.2018.2866122.  |

### Dataset Generation Strategy

For each system, two levels of scripts are maintained:

1. **Base system scripts**  
   These define the original nonlinear system equations and the corresponding Lyapunov function.

2. **Dataset generator scripts**  
   These convert symbolic expressions into tokenised machine-learning-compatible samples and generate train/valid/test datasets.

### File Mapping

#### Base system scripts
- `01.py`
- `02.py`
- `0304.py`
- `05.py`
- `06.py`
- `07.py`

#### Dataset generator scripts
- `01_dataset_generator.py`
- `02_dataset_generator.py`
- `0304_dataset_generator.py`
- `05_dataset_generator.py`
- `06_dataset_generator.py`
- `07_dataset_generator.py`
- `07_control_loop_dataset_generator.py`
  
### Common Dataset Format

Each sample follows the format:

`1| Dimension <Original function tokens> \t <Lyapunov function tokens>\n`

where:

- the **original function** consists of tokenised ODE components
- `<SPECIAL_3>` is used to separate state equations
- the **Lyapunov function** is the training target
- train / validation / test splits are generated in the dataset generator scripts
