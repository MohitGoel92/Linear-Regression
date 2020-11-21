# Supervised Learning: Linear Regression

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
  - X = Car safety features, y = Traffic accidents; examing omega to understand what makes cars safer.
  - X = Marketing budget, y = Movie revenue; examing omega to understand marketing effectiveness.

**Prediction**

- In some cases, the primary objective is to make the best prediction.
- For the machine learning function, the prediction approach will compare the real values with the predicted values.
- The focus will be on performance metrics, which measure the quality of the models predictions.
  - Performance metrics usually involve some measure of closeness between the real and prediction vales (y_p and y).
  - Without focusing on interpretability, we risk having a black-box model.
- Example of prediction exercises:
  - Interpretation: Understanding factors that may lead to customers leaving.
  - Prediction: Estimating how long customers are likely to stay can help us understand how many we still need to support, and how valuable they are to the company.

## Linear Regression

The equation below is a typical equation for a linear regression:

<p align="center"> <img width="200" src= "/Pics/W114.png"> </p>

The cost function for a linear regression is called the ***mean squared error***.

**Determining Normality**

Making our target variable normally distributed will often lead to better results. If our target variable is not normally distributed, we can apply a transformation to it and then fit our regression to predict the transformed values. There are two ways to tell if our target variable is normally distributed; we can either observe visually or use a statistical test.


