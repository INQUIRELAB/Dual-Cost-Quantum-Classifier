

# Boosting Quantum Classifier Efficiency through Data Re-Uploading and Dual Cost Functions

**Authors**: Sara Aminpour¹², Mike Banad¹, Sarah Sharif¹²*

¹School of Electrical and Computer Engineering, University of Oklahoma, Norman, OK 73019, USA  
²Center for Quantum and Technology, University of Oklahoma, Norman, OK 73019 USA  
*Corresponding author: Sarah Sharif (email: s.sh@ou.edu)


### Abstract 📜

> Quantum machine learning integrates quantum computing with classical machine learning techniques to enhance computational power and efficiency. A major challenge in Quantum machine learning is developing robust quantum classifiers capable of accurately processing and classifying complex datasets. In this work, we present an advanced approach leveraging data re-uploading, a strategy that cyclically encodes classical data into quantum states to improve classifier performance. We examine two cost functions—fidelity and trace distance—across various quantum classifier configurations, including single-qubit, two-qubit, and entangled two-qubit systems. Additionally, we evaluate four optimization techniques (L-BFGS-B, COBYLA, Nelder-Mead, and SLSQP) to determine their effectiveness in optimizing quantum circuits for both linear and non-linear classification tasks. Our results show that the choice of optimization method significantly impacts classifier performance, with L-BFGS-B and COBYLA often yielding superior accuracy. The two-qubit entangled classifier shows improved accuracy over its non-entangled counterpart, albeit with increased computational cost. Also, the two-qubit entangled classifier are the best option for real word random dataset in order to accuracy and computational cost. Linear classification tasks generally exhibit more stable performance across optimization techniques compared to non-linear tasks. Our findings highlight the potential of data re-uploading in Quantum machine learning outperforming existing quantum classifier models in terms of accuracy and robustness. This work contributes to the growing field of Quantum machine learning by providing a comprehensive comparison of classification strategies and optimization techniques in quantum computing environments, offering a foundation for developing more efficient and accurate quantum classifiers.

### About The Project 💡

This repository contains the complete Python source code for the research paper, "Boosting Quantum Classifier Efficiency through Data Re-Uploading and Dual Cost Functions." The project builds upon the foundational work of the universal quantum classifier by Pérez-Salinas et al. by introducing new cost functions, enhanced testing methodologies, and a more robust framework for evaluating classifier performance.

The code is organized into three distinct directories, representing the evolution of the project.

### Code Structure 📂

The project is divided into two main folders:

*   **`Fixed/`**: This directory represents the first major contribution of our paper. It enhances the original codebase by introducing a dual cost function system and expanded classification problems. The term "Fixed" refers to the methodology of running each experiment once with a fixed random seed for deterministic and reproducible results.

*   **`Random/`**: This directory builds upon the `Fixed` version to provide a more statistically robust analysis. The code here is designed to run each experiment multiple times (20 iterations by default) with different random initializations and then averages the performance metrics. This approach is crucial for evaluating the classifier's performance on random datasets and ensuring that the results are statistically significant and not due to a favorable random seed.

### Key Enhancements and a Comparison with the Original Code

The `Fixed` and `Random` folders introduce several key improvements over the Original codebase:

1.  **Dual Cost Functions**:
    *   We have implemented a **Trace Distance** cost function (`trace_chi`) as an alternative to the original Fidelity-based function (`fidelity_chi`). This is a core contribution of our work, explored in the new `trace_minimization.py` file.
    *   The `QuantumState.py` simulator was updated to calculate and store the Bloch vector (`self.r`), which is essential for the trace distance computation.

2.  **Expanded Problem Sets**:
    *   New classification problems, such as the `line` problem, have been added in `data_gen.py` and `problem_gen.py` to test the classifier on a wider range of linear and non-linear tasks.

3.  **Automated and Comprehensive Experimentation**:
    *   The `main.py` script has been significantly improved to automate the process of running experiments. It now systematically iterates through different classification problems (`circle`, `line`) and optimization algorithms (`l-bfgs-b`, `cobyla`, `nelder-mead`, `slsqp`), saving the results and performance metrics automatically.

4.  **Robust Statistical Analysis (in `Random/` folder)**:
    *   The most significant difference between the `Fixed` and `Random` versions lies in `big_functions.py`. The `minimizer` function in the `Random` folder is structured to run each experiment 20 times, averaging the final training and testing accuracies. This ensures that the reported performance is a reliable measure of the classifier's capabilities and is not skewed by random chance.

### How to Run the Code 🚀

1.  Ensure you have all the required libraries installed.
2.  Navigate to either the `Fixed/` or `Random/` directory based on your desired testing methodology.
3.  Open the `main.py` file.
4.  Modify the parameters at the top of the file to configure your experiment:
    *   `qubits`: Number of qubits (e.g., 1 or 2).
    *   `layers`: Number of data re-uploading layers.
    *   `chi`: The cost function to use (`'fidelity_chi'` or `'trace_chi'`).
    *   `entanglement`: Use entanglement (`'y'` or `'n'`).
    *   `problem`: A list of problems to test (e.g., `['circle', 'line']`).
    *   `method`: A list of optimization algorithms to use.
5.  Run the script from your terminal:
    ```bash
    python main.py
    ```
6.  Results, including performance summaries and plots, will be saved in organized subdirectories.

### Requirements 💻

*   Python 3.x
*   NumPy
*   Matplotlib
*   SciPy
*   Scikit-learn

### Citation Information ✍️

If you use the code from the `Fixed` or `Random` folders in your research, please cite our paper:

```bibtex
@article{aminpour2024boosting,
  title={Boosting Quantum Classifier Efficiency through Data Re-Uploading and Dual Cost Functions},
  author={Aminpour, Sara and Banad, Mike and Sharif, Sarah},
  journal={arXiv preprint arXiv:2405.09377},
  year={2024},
  eprint={2405.09377},
  archivePrefix={arXiv},
  primaryClass={quant-ph}
}
```

If you use code from the original code or build upon the foundational concepts, please also cite the original work:

```bibtex
@article{P_rez_Salinas_2020,
   title={Data re-uploading for a universal quantum classifier},
   volume={4},
   ISSN={2521-327X},
   url={http://dx.doi.org/10.22331/q-2020-02-06-226},
   DOI={10.22331/q-2020-02-06-226},
   journal={Quantum},
   publisher={Verein zur Forderung des Open Access Publizierens in den Quantenwissenschaften},
   author={Pérez-Salinas, Adrián and Cervera-Lierta, Alba and Gil-Fuster, Elies and Latorre, José I.},
   year={2020},
   month={Feb},
   pages={226}
}
