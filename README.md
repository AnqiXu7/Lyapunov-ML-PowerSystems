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
