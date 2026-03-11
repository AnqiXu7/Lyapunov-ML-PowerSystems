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

| ID | System Description | Dimension | Equation Structure | Lyapunov Function | Source |
|----|--------------------|-----------|--------------------|-------------------|--------|
| 01 | Nonlinear multi-machine swing-equation system | 6D | Trigonometric coupling between rotor angles and frequencies | Quadratic energy-type Lyapunov function | Based on power system transient stability models |
| 02 | Reduced two-state nonlinear swing system | 2D | Angle–frequency nonlinear dynamics with sin coupling | Quadratic Lyapunov function with trigonometric terms | Classical transient stability benchmark |
| 03 | Polynomial recast power system stability model | 4D | Polynomial differential-algebraic representation of power system dynamics | SOS-based polynomial Lyapunov function | Anghel et al., Algorithmic Construction of Lyapunov Functions |
| 04 | Two-machine infinite-bus power system model | 4D | Nonlinear swing equations expressed in shifted coordinates | Polynomial Lyapunov function constructed via SOS optimisation | Anghel et al., 2013 |
| 05 | Synthetic nonlinear benchmark system | 3D | Trigonometric nonlinear coupled ODE system | Quadratic Lyapunov candidate | Synthetic dataset constructed in this project |
| 06 | Double-machine infinite-bus system | 4D | Nonlinear swing equations with rational Lyapunov structure | Rational Lyapunov function | Han et al., Optimal Rational Lyapunov Functions |
| 07 | Coupled nonlinear benchmark system | 3D | Nonlinear coupled ODE dynamics | Quadratic Lyapunov function | Synthetic dataset constructed in this project |
| 08 | Virtual Synchronous Generator transient stability model | 2D | Nonlinear inverter swing dynamics | Lyapunov function derived for VSG stability analysis | VSG transient stability literature |

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
- `08.py`

#### Dataset generator scripts
- `01_dataset_generator.py`
- `02_dataset_generator.py`
- `0304_dataset_generator.py`
- `05_dataset_generator.py`
- `06_dataset_generator.py`
- `07_dataset_generator.py`
- `08_dataset_generator.py`
- `08_control_loop_dataset_generator.py`
  
### Common Dataset Format

Each sample follows the format:

`1| <Original function tokens> \t <Lyapunov function tokens>\n`

where:

- the **original function** consists of tokenised ODE components
- `<SPECIAL_3>` is used to separate state equations
- the **Lyapunov function** is the training target
- train / validation / test splits are generated in the dataset generator scripts
