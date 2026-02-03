[README.md](https://github.com/user-attachments/files/25052598/README.md)
# Logistic Regression from Scratch

A NumPy-based implementation of logistic regression with advanced optimization techniques, built without relying on scikit-learn or PyTorch/TensorFlow.

## Overview

This project demonstrates a complete implementation of **logistic regression** using only fundamental Python libraries:
- **NumPy** - Matrix operations and fast numerical computations
- **Matplotlib** - Training curves visualization
- **tqdm** - Progress tracking during training loop

## 🎯 Key Features

### Core Implementation
✅ **Binary Classification** - Multi-dimensional input support
✅ **Sigmoid Activation** - Numerically stable implementation with optimized conditions  
✅ **Cross-Entropy Loss** - Binary cross-entropy (logloss) function  
✅ **Gradient Descent** - Backpropagation with weight and bias updates

### Advanced Techniques
🔧 **L2 Regularization** - Prevents overfitting via weight penalty
📉 **ReduceLROnPlateau Scheduler** - Reduces learning rate by 10% when accuracy plateaus  
🛑 **Early Stopping** - Halts training when accuracy stops improving (patience-based)

### Data Pipeline
📊 **Data Scaling** - StandardScaler
🔀 **Train-Test Split** - 80/20 stratification  

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **92.0%** |
| **Test Loss (Logloss)** | **0.191** |
| **Training Accuracy** | 100.0% |
| **Training Loss** | 0.0652 |
| **Total Epochs** | 1,001 / 2,000 (early stopped) |
| **Final Learning Rate** | 0.0001 |

<img width="1233" height="547" alt="image" src="https://github.com/user-attachments/assets/a7f88b06-5728-4d9f-8de1-8902085ec0c0" />

---

## 🔬 Mathematical Details

### **Sigmoid Function**
$$\sigma(z) = \begin{cases}
\frac{1}{1 + e^{-z}} & \text{if } z \geq 0 \\
\frac{e^z}{1 + e^z} & \text{if } z < 0
\end{cases}$$

*Numerically stable implementation prevents overflow/underflow.*

### **Binary Cross-Entropy Loss**
$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right]$$

### **Gradient Descent Update**
```math
w := w - \alpha \cdot \frac{\partial \mathcal{L}}{\partial w} + \text{ridge\_gradient}(\alpha, w)
```
```math
b := b - \alpha \cdot \frac{\partial \mathcal{L}}{\partial b}
```
### **L2 Regularization Penalty**
(\text{ridge_gradient} = \frac{\alpha \cdot w}{n})


### **ReduceLROnPlateau**
- Monitors accuracy every 5 epochs
- If accuracy doesn't improve for 21% of total epochs, multiply learning rate by 0.1
- Minimum learning rate: 1e-6 (prevents excessive shrinkage)

### **Early Stopping**
- Activates after 50% of epochs
- Tracks accuracy stagnation with configurable patience
- Stops training when patience counter reaches zero

---

## 🚀 Quick Start

### Installation
```bash
pip install numpy matplotlib tqdm
```

---

## 📊 Training Dynamics

### Loss Curve
The training loss decreases significantly in early epochs, stabilizing as the model approaches optimal weights. With the ReduceLROnPlateau scheduler, learning rate reductions prevent oscillations near the minimum.

### Accuracy Progression
- **Early Phase (Epochs 0–300):** Rapid accuracy improvement from ~50% → 91.25%
- **Mid Phase (Epochs 300–600):** Steady progress toward 99%+
- **Late Phase (Epochs 600+):** Plateau detection triggers learning rate reduction
- **Final (Epoch 1,001):** **Early stopping activates** with 100% training accuracy

---

## 🛠️ Implementation Highlights

### Numerically Stable Sigmoid
```python
def sigmoid(z):
    return np.where(
        z >= 0, 
        1 / (1 + np.exp(-z)),           # Standard formula (z ≥ 0)
        np.exp(z) / (1 + np.exp(z))     # Numerically stable (z < 0)
    )
```

### Gradient Computation
```python
dw = (X.T @ (raw_probs - y)) / len(y)     # Weight gradient
db = np.mean(raw_probs - y)                # Bias gradient
```

### L2 Regularization
```python
if alpha:
    w = w - lr * dw + ridge_gradient(alpha, w)
else:
    w = w - lr * dw
```

### Learning Rate Scheduler
```python
def reduce_on_plateau(lr, factor=0.1, stop=1e-6):
    '''Reduce Learning Rate on Plateau'''
    if lr > stop:
        return lr * factor
    return lr
```

### Learning Rate Scheduler
Monitors accuracy every 5 epochs. Realization of this algorithm is not clear enough, but I will work on improving it.

### Early Stopping
Compares consecutive epoch accuracies. If stagnation detected, training halts immediately.

---

## 📝 Dataset Details

### Synthetic Data Generation
- **Sample Size:** 50,000 features (you may play with this parameter) reshaped to (500 samples, 100 features)
- **Target Variable:** Binary classification based on feature mean
  - Label = 1 if mean(features) > 50
  - Label = 0 otherwise
- **Train-Test Split:** 80% training, 20% test

### Preprocessing
StandardScaler normalization:
- Mean and std computed on training set only
- Applied identically to test set
- Prevents data leakage

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **NumPy** | ≥1.20 | Core numerical operations |
| **Matplotlib** | ≥3.3 | Visualization |
| **tqdm** | ≥4.50 | Progress bars |

---

## 🎓 Learning Outcomes

This implementation demonstrates:
- ✅ Fundamental ML algorithm from scratch
- ✅ Numerical stability in floating-point operations
- ✅ Regularization techniques (L2)
- ✅ Learning rate scheduling for better convergence
- ✅ Early stopping to prevent overfitting
- ✅ Proper train-test data handling
- ✅ Vectorized NumPy operations for efficiency
