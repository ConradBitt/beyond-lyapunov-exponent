# Beyond the largest lyapunov exponent: characterizing periodic attractor robustness and crisis dynamics in coupled Hénon maps

This repository contains the scripts and numerical simulation code used in the study of the robustness of periodic attractors in coupled nonlinear systems. The main focus is to investigate the limitations of the **largest Lyapunov exponent** in characterizing dynamics at crisis boundaries.

**Key Features:**
*   **Model:** Implementation of two symmetrically coupled Hénon maps.
*   **Implemented Analyses:**
    *   Calculation of **Finite Time Lyapunov Exponents** (largest and transversal).
    *   **Periodicity** detection and **escape time** statistics.
    *   Calculation of **Basin Stability** and **Basins entropies**.
    *   Estimation of **uncertainty exponent** and **analysis of basin geometry** (including “riddled” basins).
*   **Objective:** To reproduce the results demonstrating how periodic structures embedded in crisis regions exhibit transverse instability and severe loss of predictability, requiring metrics beyond the standard Lyapunov exponent.

**How to Cite:**
If you use this code in your research, please cite the associated article:
*[Insert Full Article Reference Here]*

## **License:**
This project is licensed under the **MIT** license. Feel free to use, modify, and distribute the code.

MIT License

Copyright (c) 2026 Conrado Ferreira Bittencourt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Environment Setup and Configuration

To ensure the reproducibility of the simulations, this repository includes the `environment.yml` file, which defines all necessary dependencies. Follow the steps below to clone the repository and set up the environment:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ConradBitt/beyond-lyapunov-exponent.git
    cd beyond-lyapunov-exponent
    ```

2.  **Create the Conda environment:**
    Run the following command to create an isolated environment and install the listed dependencies:
    ```bash
    conda env create -f environment.yml
    ```

3.  **Activate the environment:**
    ```bash
    conda activate beyond-lyapunov
    ```

4.  **Execution:**
    With the environment active, the simulation and analysis scripts are ready to run.

If you need to update the environment after making changes to the YAML file, use the command `conda env update -f environment.yml --prune`.
