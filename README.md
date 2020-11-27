# Linear Regression

There are different types of machine learning, they include:

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

**Interpretation** 

- In some cases, the primary objective is to train a model to find insights from the data.
- For the machine learning function given above, the interpretation approach uses omega to give us insight into a system.
- Common workflow:
  - Gather X, Y: Train the model by finding the omega that gives the best predictions.
  - Focus on omega rather than the predicted values to generate insights.
- Example of interpretation exercises:
  - X = Customer demographics, y = Sales data; examine omega to understand the loyalt by segment.
  - X = Car safety features, y = Traffic accidents; examine omega to understand what makes cars safer.
  - X = Marketing budget, y = Movie revenue; examine omega to understand marketing effectiveness.

**Prediction**

- In some cases, the primary objective is to make the best prediction.
- For the machine learning function, the prediction approach will compare the real values with the predicted values.
- The focus will be on performance metrics, which measure the quality of the models predictions.
  - Performance metrics usually involve some measure of closeness between the real and prediction vales (y_p and y).
  - Without focusing on interpretability, we risk having a black-box model.
- Example of prediction exercises:
  - Interpretation: Understanding factors that may lead to customers leaving.
  - Prediction: Estimating how long customers are likely to stay may help us understand how many we still need to support, and how valuable they are to the company.

## Linear Regression

A linear regression models the relationship between a continuous variable and one or more scaled variables. It is usually represented as a dependent function equal to the sum of a coefficient plus scaling factors times the independent variables. The equation below is a typical equation for a linear regression:

<p align="center"> <img width="200" src= "/Pics/W114.png"> </p>

Residuals are defined as the difference between an actual value and a predicted value. The cost function for a linear regression is called the ***mean squared error***. In addition, common measures of error include the Sum of Squared Error (SSE), Total Sum of Squares (TSS), and Coefficient of Determination (R2). 

**Determining Normality**

Making our target variable normally distributed will often lead to better results. If our target variable is not normally distributed, we can apply a transformation to it and then fit our regression to predict the transformed values. There are two ways to tell if our target variable is normally distributed; we can either observe visually or use a statistical test.

The statistical test will test whether a distribution is normally distributed or not:
 - The test outputs a p-value. The higher the p-value is the closer the distribution is to a normal distribution.
 - We accept that the distribution is normal if p > 0.05.

**Box Cox**

The Box Cox transformation is a parameterised transformation that tries to get distributions "as close to a normal distribution as possible".

It is defined as:

<p align="center"> <img width="200" src= "/Pics/w115.png"> </p>

The square root uses the exponent of 0.5 (or 1/2), but Box Cox lets its exponent vary so it can find the best one. We will first use the Box Cox transformation on the data set, and then use the inverse transformation. The codes below are the codes that have been used in the Jupyter notebook.

```
# Applying the Box Cox transformation
from scipy.stats import boxcox

boxcox_result = boxcox(y_train)
y_train_boxcox = boxcox_result[0]
lam = boxcox_result[1]

lr.fit(X_train, y_train_boxcox)
y_pred_boxcox = lr.predict(X_test)
```

```
# Applying the inverse Box Cox transformation
from scipy.special import inv_boxcox

y_pred = inv_boxcox(y_pred_boxcox, lam)
```

**For the Jupyter notebook, please see:** https://github.com/MohitGoel92/Linear-Regression/tree/main/Normally%20Distributing%20Variables%20for%20Regression

# Regularisation Techniques

## The Bias-Variance Trade-Off

The diagram below illustrates the relationship between model complexity and error. For Jtrain, the error reduces as the complexity increases. However, for Jcv the error reduces as complexity increases to a certain point, but starts to increase after this particular point. This is due to Jtrain overfitting the dataset. If we overfit the dataset, the model may accurately predict the dataset on which it was trained on however, it is likely to be a poor fit on a new dataset. This is demonstrated by the increasing error for Jcv after the model has reached a particular complexity.

<p align="center"> <img width="600" src= "/Pics/W31.png"> </p>

There are 3 sources of model error, they are:
 - Bias: Being wrong
 - Variance: Being unstable
 - Irreducible error: Unavoidable randomness

**Tendency:** The expectation of out-of-shape behaviour over many training set samples.

**Bias:** The tendency of predictions to miss true values. This is worsened by missing information and overly-simplistic assumptions. A common reason for higher bias is underfitting, therefore missing real patterns in the data.

**Variance:** The tendency of predictions to fluctuate or be inconsistent. This is characterised by sensitivity or output to small changes in inpute data. A common reason for higher variance is overly complex or poorly fit models.

**Irreducible Error:** The tendency to instrinsic uncertainty/randomness. This is present in even the best possible models.

The diagram below is a visual representation of bias and variance. From the diagram, we observe that variance indicates how far spread the predictions are, and bias refers to how close the predictions are to the real values. Ideally we want to be highly consistent for predictions that are close to perfect on average.

<p align="center"> <img width="550" src= "/Pics/W332.png"> </p>

For the graphs below, the blue curve represents the true model and the black line/curve represents the alternative models for explanatory purposes.

<p align="center"> <img width="1000" src= "/Pics/W32.png"> </p>

Polynomial Degree = 1: 
- High bias but low variance.
- Poor at both training and predicting.

Polynomial Degree = 4: 
- Bias and variance is just right.
- Training and predicting is just right.

Polynomial Degree = 15: 
- Low bias but high variance.
- Good at training but poor at predicting.

## Bias-Variance Trade-off Visualised

<p align="center"> <img width="600" src= "/Pics/W34.png"> </p>

The diagram above summarises the key points below:

- Model adjustments that decrease bias often increase variance, and vice versa.
- The bias-variance is analogous to a complexity trade-off.
- Finding the best model means choosing the right level of complexity.
- Ideally, we want a model elaborate enough to not underfit, but not so exceedingly elaborate that it overfits.
- The higher the degree of a polynomial regression, the more complex the model (lower bias, higher variance).
- At lower degrees, we observe visual signs of bias and the predictions are too rigid to capture the curve pattern in the data.
- At higher degrees, we see visual signs of variance. Predictions fluctuate wildly because of the models sensitivity.
- The goal is to find the right degree, such that the model has sufficient complexity to describe the data without overfitting.

## Regularisation and Model Selection

The function below is the **Adjusted Cost Function**:

<p align="center"> <img width="200" src= "/Pics/W35.png"> </p>

where
 - M(w): Model Error
 - R(w): Function of Estimated Parameter(s)
 - λ: Regularisation Strength Parameter (Lambda)

Regularisation adds an (adjustable) regularisation strength parameter directly into the cost function. The lambda (λ) adds a penalty proportional to the size of the estimated model parameter, or a function of the parameter. The larger the λ, the more we will penalise stronger parameters and the less complex our model will be as we try to minimise our function. Increasing the cost function controls the amount of penalty.

The regularisation strength parameter λ allows us to manage the complexity tradeoff:
 - More regularisation introduces a simpler model or more bias (i.e. Higher λ => stronger penalty => simpler model).
 - Less regularisation makes the model more complex and increases variance (i.e. Lower λ => weaker penalty => more complex model).
 
If our model has overfit (variance is too high), regularisation can improve the generalisation error and reduce variance (better generalised fit).

 ## Regularisation and Feature Selection
 
Regularisation performs feature selection by shrinking the contribution of features. For L1 - regularisation, this is accomplished by driving some coefficients to zero. 

**Note:** Feature selection can also be performed by removing features by *Principal Component Analysis* or *Linear Discriminant Analysis*.

Reducing the number of features may prevent overfitting. For some models, a reduced number of features will improve fitting time and/or results. Therefore, identifying the most critical features can improve model interpretability.

## Ridge Regression (L2)

**Note:** When fitting the linear regression to the dataset, feature scaling is not a necessary step as the linear_model library takes care of this for us. However, when dealing with Ridge regression feature scaling is now required.

The Ridge regression cost function is given below:

<p align="center"> <img width="350" src= "/Pics/W39.png"> </p>

where RSS is the *Residual Sum of Squares*.

- The complexity penalty λ is applied proportionally to the squared coefficient values. This increases/decreases the effect of the square of each of the coefficient values.
- The penalty terms has the effect of "shrinking" coefficients toward 0.
- This imposes bias on the model, but also reduces variance.
- We can select the best regularisation strength λ via cross-validation.
- It's best practice to scale features (i.e. using StandardScaler) so penalties aren't impacted by variable scale.

Given below is a new cost function that we will be using for Ridge regression. This has been derived from taking the original linear regression cost function and adding on a penalty to reduce complexity of our model.

<p align="center"> <img width="450" src= "/Pics/W311.png"> </p>

**Note:**

- The penalty shrinks the magnitude of all coefficients.
- Larger coefficients are strongly penalised because of squaring. For instance, a coefficient of 2 will be penalised 4 times as much as a coefficient of 1, and a coefficient of 3 will be penalised 9 times as much.

The below graphs illustrate the behavioural changes of the polynomials as λ changes.

<p align="center"> <img width="1000" src= "/Pics/W312.png"> </p>

The diagram below demonstrates how the coefficients shrink as the penalty parameter λ increases.

<p align="center"> <img width="500" src= "/Pics/W313.png"> </p>

**Complexity Tradeoff:** Variance reduction may outpace the increase in bias, leading to a better model fit. This is depicted by the figure below.

<p align="center"> <img width="500" src= "/Pics/W314.png"> </p>

**Note:** Ridge produces a computationally efficient algorithm in comparison to Lasso.

## LASSO Regression (L1)

**LASSO:** Least Absolute Shrinkage and Selection Operator.

The only difference between LASSO and Ridge regression is how we penalise the cost function using our coefficients. With Ridge (L2), we use the coefficient squared, and with LASSO we'll be using the absolute value of each one of these coefficients.

<p align="center"> <img width="350" src= "/Pics/W315.png"> </p>

**Note:** Penalties are closely related to L1/L2 norms that measure vector length.

In LASSO regression, the complexity penalty λ is proportional to the absolute value of the coefficients. In Ridge regression, it's proportional to the square of the coefficients. LASSO has the similar effect to Ridge in terms of complexity tradeoff, in that "Increasing λ raises bias but lowers variance".

**Note:** LASSO is more likely than Ridge to perform feature selection. For instance, for a fixed λ LASSO is more likely to result in coefficients being set to zero.

Given below is the cost function that we will be using for LASSO regression. 

<p align="center"> <img width="450" src= "/Pics/W316.png"> </p>

**Note:**

- The penalty selectively shrinks some coefficients.
- LASSO can be used for feature selection.
- In comparison to Ridge regression, LASSO is slower to converge.
- LASSO will eliminate features quickly if we have a high value for lambda (λ).

The diagram below demonstrates how shrinkage and selection effect as regularisation strength increases. We observe how some features quickly drop to 0.

<p align="center"> <img width="450" src= "/Pics/W317.png"> </p>

**Complexity Tradeoff:** Variance reduction may outpace the increase in bias, leading to a better model fit. This can be observed by the diagram below.

<p align="center"> <img width="450" src= "/Pics/W318.png"> </p>

## Elastic Net: Between Ridge and LASSO

**Elastic Net:** The hybrid approach.

The cost function is given below:

<p align="center"> <img width="225" src= "/Pics/W319.png"> </p>

**Notes:**

- Validation gives us an empirical (observable) method for selecting between different models.
- LASSO's feature selection property yields an interpretability bonus, but may underperform if the target truly depends on many of the features.
- Elastic Net, an alternative hybrid approach introduces a new parameter α (alpha) that determines a weighted average of L1 and L2 penalties.
- Ridge regression is a computationally efficient algorithm and penalises certain weights even higher.

From the below we observe how Elastic Net combines penalties from both Ridge and LASSO regression.

<p align="center"> <img width="550" src= "/Pics/W320.png"> </p>

**Note:** It requires tuning of an additional parameter that determines emphasis of L1 Vs L2 regularisation penalties.

The below graphs illustrate the behavioural changes of the polynomials as λ1 and λ2 change.

<p align="center"> <img width="1000" src= "/Pics/W322.png"> </p>

**Note:** λ1 and λ2 may or may not be equal.

## Recursive Feature Elimination (RFE)

This approach combines:

- A model or estimation approach.
- A desired number of features.

RFE then repeatedly applies the model, measures feature importance and recursively removes less important features.

Example Code:

```
from sklearn.feature_selection import RFE, RFECV

rfe = RFE("classifier or model", n_features_to_select = 5)
rfe = rfe.fit(X_train, y_train)
y_pred = rfe.predict(X_test)
```
The RFECV class will perform feature elimination using cross validation.

## Polynomial Features and Regularisation: Intuitive Analysis and Further Details

