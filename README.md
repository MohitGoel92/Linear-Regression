# Supervised Learning: Linear Regression

There are different types of machines learning, they include:
- Supervised: When the data points have a known outcome.
- Unsupervised: When the data points have an unknown outcome.
- Semi-Supervised: When we have data with known outcomes and data without outcomes.

The equation below represents a machine learning function:


<p align="center"> <img width="200" src= "/Pics/w113.png"> </p>

where
  - X: Inpute
  - Y_p: Output (values predicted by the model)
  - f(.): Prediction function that generates predictions from x and omega
  
**Observations:** The rows or examples/samples the model will see.
**Features:** The different ways that we measure each observation (variables that may or may not influence the target variable).

- A single observation can be represented by a row.
- A single feature or variable can be represented by a column.
- A hyperparameter is a parameter that is not learned directly from the data, but relates to implementation; training our machine learning model.
- Fit parameters involve aspects of the model we estimate (fit) using the data.
- Regression is when we predict a numeric value.
- Classification is when we predict a categorical value.
- The loss measures how close our predictions are to the true values.
- We use features X and the outcome Y, to choose parameters alpha to minimise the loss.

## Interpretation and Prediction

