

# Boosting Quantum Classifier Efficiency through Data Re-Uploading and Dual Cost Functions

**Authors**: Sara Aminpour¹², Mike Banad¹, Sarah Sharif¹²*

¹School of Electrical and Computer Engineering, University of Oklahoma, Norman, OK 73019, USA  
²Center for Quantum and Technology, University of Oklahoma, Norman, OK 73019 USA  
*Corresponding author: Sarah Sharif (email: s.sh@ou.edu)


### Abstract 📜

> Quantum machine learning integrates quantum computing with classical machine learning techniques to enhance computational power and efficiency. A major challenge in Quantum machine learning is developing robust quantum classifiers capable of accurately processing and classifying complex datasets. In this work, we present an advanced approach leveraging data re-uploading, a strategy that cyclically encodes classical data into quantum states to improve classifier performance. We examine two cost functions—fidelity and trace distance—across various quantum classifier configurations, including single-qubit, two-qubit, and entangled two-qubit systems. Additionally, we evaluate four optimization techniques (L-BFGS-B, COBYLA, Nelder-Mead, and SLSQP) to determine their effectiveness in optimizing quantum circuits for both linear and non-linear classification tasks. Our results show that the choice of optimization method significantly impacts classifier performance, with L-BFGS-B and COBYLA often yielding superior accuracy. The two-qubit entangled classifier shows improved accuracy over its non-entangled counterpart, albeit with increased computational cost. Also, the two-qubit entangled classifier are the best option for real word random dataset in order to accuracy and computational cost. Linear classification tasks generally exhibit more stable performance across optimization techniques compared to non-linear tasks. Our findings highlight the potential of data re-uploading in Quantum machine learning outperforming existing quantum classifier models in terms of accuracy and robustness. This work contributes to the growing field of Quantum machine learning by providing a comprehensive comparison of classification strategies and optimization techniques in quantum computing environments, offering a foundation for developing more efficient and accurate quantum classifiers.

### About The Project 💡

This repository contains the complete Python source code for the research paper, "Boosting Quantum Classifier Efficiency through Data Re-Uploading and Dual Cost Functions." 
The code is organized into two distinct directories, representing the evolution of the project.

### Code Structure 📂

The project is divided into two main folders:

*   **`Fixed/`**: This directory represents the first major contribution of our paper. It enhances the introductory of the dual cost function system and expanded classification problems. The term "Fixed" refers to the methodology of running each experiment once with a fixed random seed for deterministic and reproducible results. More detail of dual cost function system can be found in the supplementary note 8.

*   **`Random/`**: This directory represent random dataset to provide more statistically robust analysis. In order to represent a fundamental dataset for real application each run has been repeated multiple times with different random initializations (20 iterations by default). The result of each set of iteration has been averaged to show the performance metrics. This approach is crucial for evaluating the classifier's performance on random datasets and ensuring that the results are statistically significant and not due to a favorable random seed.


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

