
# Handwritten Digit Recognition: Classical Models vs. ANN with Gradio Deployment

## Overview

This project presents a comprehensive study comparing **classical machine learning models** (Logistic Regression, Random Forest, SVM with PCA) and a **deeply tuned Artificial Neural Network (ANN)** for handwritten digit classification using the **MNIST** dataset.  
We further demonstrate a **Gradio web application** for real-time interaction with the trained ANN model.

Our main contributions:
- Built and tuned classical models using **PCA** for dimensionality reduction.
- Designed and optimized an **ANN using Keras and Keras Tuner**.
- Evaluated all models based on **accuracy, macro F1-score, confusion matrices, training and inference times**.
- Deployed the best-performing ANN model via a **Gradio web interface** for live predictions.

## Tools and Technologies Used

| Category                 | Tool / Library                                  |
|--------------------------|-------------------------------------------------|
| Programming Language     | Python 3.x                                      |
| Data Science Libraries   | NumPy, Pandas, scikit-learn                     |
| Deep Learning Framework  | TensorFlow / Keras                              |
| Visualization Tools      | Matplotlib, Seaborn                             |
| Hyperparameter Tuning    | Keras Tuner                                     |
| Deployment Interface     | Gradio                                          |
| Evaluation Metrics       | scikit-learn (accuracy, F1-score, confusion matrix) |
| IDE                      | Jupyter Notebook, VS Code                       |
| File Format              | `.ipynb`, `.npy`, `.h5`, `.mov`                 |


## Dataset Description: MNIST Handwritten Digits

- **Dataset Name:** [MNIST](http://yann.lecun.com/exdb/mnist/)
- **Source:** Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- **Type:** Supervised classification dataset
- **Format:** 28×28 grayscale images
- **Classes:** 10 (digits 0 to 9)
- **Training Samples:** 60,000
- **Test Samples:** 10,000
- **Total Size:** ~11.6 MB

### Why MNIST?
MNIST is widely used for benchmarking image classification models. Its relatively small size makes it ideal for prototyping, yet the digit variety provides enough complexity to meaningfully compare classical vs deep learning models.



## Results Summary

| Model | Accuracy | Macro F1-Score | Training Time | Inference Speed | Notes |
|:---|:---:|:---:|:---:|:---:|:---|
| Logistic Regression + PCA | ~92.2% | ~0.92 | ~3 hours | Slow | High training cost |
| Random Forest + PCA | 97.03% | 0.9701 | 112 seconds | Fast | Best classical model |
| SVM (RBF Kernel) + PCA | ~97.1% | ~0.970 | ~1.5 hours | Moderate | Accurate but slow to retrain |
| ANN (Baseline) | 97.0% | ~0.970 | ~60 seconds | Fast | Good starting point |
| ANN (Tuned with Keras Tuner) | **99.0%** | **0.989** | ~180 seconds | Very Fast | Best overall model |

The ANN with hyperparameter tuning achieved the highest performance without requiring PCA or manual feature engineering.


## Project Structure

```
Project_3/
│
├── train.ipynb                  # Notebook for ANN training
├── test.ipynb                   # Notebook for testing/evaluation
├── Project_3.docx               # Final project report 
│
├── data/                        # Contains input datasets
│   ├── x_train.npy
│   ├── y_train.npy
│   ├── x_test.npy
│   └── y_test.npy          
│
├── Saved_models/                # Stores trained model files
│   └── ann_mnist.h5
│
├── Competition/                 # Folder for competition variants
│   └── ...

```

## Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/handwritten-digit-classification.git
cd handwritten-digit-classification
```

2. **Install dependencies**
```bash
pip install tensorflow scikit-learn gradio matplotlib numpy
```

3. **Run ANN training**
```bash
python train.ipynb
```

4. **Evaluate model performance**
```bash
python test.ipynb
```

5. **Launch Gradio app**
```bash
python test.ipynb
```
You can draw a digit, or pick a test sample!


## Evaluation Metrics

- **Test Accuracy:** 99.0% (Tuned ANN)
- **Macro F1-Score:** 0.989
- **Confusion Matrix:** Strong diagonal dominance across all digits.
- **Training Time:** ~180 seconds (on GPU)
- **Inference Time:** <100ms per sample


## Key Learnings

- **PCA** dramatically reduces feature space, enabling faster classical model training.
- **Random Forest** with PCA provides a strong baseline for classical approaches.
- **Tuned ANN** surpasses classical models in accuracy without needing dimensionality reduction.
- **Gradio** makes deploying ML models accessible and interactive.

## Demo video:

**[Watch Demo Video](https://drive.google.com/file/d/15xpqgpW3eiNuCdEVuv-WH4WbGLWTtt3Q/view?usp=sharing)**


## Future Work

- Extend to **CNN architectures** (e.g., LeNet, ResNet) for even higher accuracy.
- Implement **quantized models** for edge deployment.
- Train models on **augmented datasets** (rotation, noise) for robustness.

## Authors

- Swetha Gendlur Nagarajan
- Masters in Applied Data Science in University of Florida.


