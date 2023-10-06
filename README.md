# Intro-to-ML

Instructor : Prof. Yen-Yu Lin

Semester : 2023 Fall

## HW1 - Linear Regression
- Closed-form Solution :
$$\hat{\beta} = (X^T X)^{-1}X^T Y$$

- Gradient Descent : 
$$\theta ^{*} = \theta ^n - \eta \frac{\partial L}{\partial \theta} \bigg|_{\theta = \theta ^n} $$

- MSE (Mean Square Error) as loss function :
$$MSE = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y_i})^2$$
