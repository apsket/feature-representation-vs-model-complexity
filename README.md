# Learning Decision Boundaries: Representation, Inductive Bias, and Model Complexity

## Motivation

In many machine learning problems, performance is often attributed to model complexity. Non-linear datasets are typically handled with flexible models such as kernel methods or neural networks.

This project explores an alternative perspective:

> Complex decision boundaries can often be recovered by combining simple models with structured representations of the data.

---

## Approach

We study a non-linearly separable classification problem and compare three strategies:
![Raw data](results/figures/raw_data.png)

### 1. Flexible Model (Baseline)
- Support Vector Machine with RBF kernel.
- High-capacity model that learns non-linear boundaries directly.

### 2. Feature-Engineered Linear Model
- Logistic regression applied to transformed features.
- Structured features capturing geometry (radius and angular components).

### 3. Non-Linear Logistic Regression
- Polar logistic regression.

### 4. Parametric Geometric Boundary
- Custom decision boundary inspired by conic sections.
- Boundary defined analytically and learned via numerical optimization.
- Includes both fixed and learnable orientation.

---

## Key Idea

Instead of increasing model complexity, we modify the **representation of the data**:

- Encode geometric structure explicitly.
- Transform to coordinate systems aligned with the problem.
- Reduce the burden on the learning algorithm.

This shifts complexity from the model to the representation.

![Raw data](results/figures/polar_data_rectangular.png)

---

## Results

- Feature-engineered logistic regression achieved performance comparable to kernel SVM on structured non-linear data.
- Parametric geometric models provided interpretable boundaries with competitive accuracy.
- Flexible models (SVM) were more robust under increasing noise.
- Structured representations worked well when aligned with the underlying geometry, but degraded when misaligned.

---

## Insights

### Representation vs Model Complexity

- Model performance is not only determined by algorithm choice, but by how the data is represented.
- Simple models can recover complex behavior when structure is encoded explicitly.

### Inductive Bias Matters

- Feature engineering introduces assumptions about symmetry and geometry.
- These assumptions can significantly improve efficiency—but also introduce failure modes.

### Trade-offs

- Flexible models: robust, but less interpretable.
- Engineered representations: efficient and interpretable, but require insight.
- Parametric models: highly structured, but sensitive to assumptions.

## Extensions

To further test these ideas, the project includes experiments on synthetically generated datasets with controlled noise and known decision boundaries. This allows evaluation of:

- Robustness to noise
- Stability of different modeling approaches
- When feature engineering breaks down relative to more flexible models

---

## Repository Structure

- `data/` — input datasets (synthetic and real)
- `notebooks/` — exploratory analysis
- `src/` — model implementations
- `results/` — visualizations and comparisons  

---

## Summary

This project demonstrates that:

> Learning performance depends as much on **representation and structure** as on model complexity.

Understanding this trade-off is essential when designing systems that must operate under constraints such as interpretability, efficiency, or limited data.
