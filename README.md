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

### 1. Dataset Categories by Source

#### Literature-based benchmark systems
These datasets are derived from published power system stability models and reference papers.

- **01**: Kundur 3-generator reduced model
- **02**: Classical 2-bus system
- **03/04**: Multi-machine benchmark models based on literature-reported nonlinear dynamics
- **05**: 3-state nonlinear system
- **06**: Double-machine versus infinite bus system with rational Lyapunov function
- **07**: 3-state nonlinear coupled system
- **08**: Virtual Synchronous Generator (VSG) transient stability model

### 2. Dataset Categories by Mathematical Structure

#### Trigonometric nonlinear systems
Most systems contain nonlinear trigonometric terms such as `sin(.)` and `cos(.)`, which are commonly found in power system swing dynamics and energy-function-based stability analysis.

Included systems:
- 01
- 02
- 03/04
- 05
- 07
- 08

#### Rational Lyapunov systems
Some systems use rational Lyapunov functions, introducing more complex symbolic structures.

Included systems:
- 06

#### Control-loop-augmented systems
These extend the base nonlinear system by adding additional control dynamics.

Included systems:
- 08_control_loop

### 3. Dataset Categories by State Dimension

#### 2-state systems
- 02
- 08

#### 3-state systems
- 05
- 07
- 08_control_loop

#### 4-state systems
- 03/04
- 06

#### 6-state systems
- 01

### 4. Dataset Generation Strategy

For each system, two levels of scripts are maintained:

1. **Base system scripts**  
   These define the original nonlinear system equations and the corresponding Lyapunov function.

2. **Dataset generator scripts**  
   These convert symbolic expressions into tokenised machine-learning-compatible samples and generate train/valid/test datasets.

### 5. File Mapping

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
  
### 6. Common Dataset Format

Each sample follows the format:

`1| <Original function tokens> \t <Lyapunov function tokens>\n`

where:

- the **original function** consists of tokenised ODE components
- `<SPECIAL_3>` is used to separate state equations
- the **Lyapunov function** is the training target
- train / validation / test splits are generated in the dataset generator scripts
