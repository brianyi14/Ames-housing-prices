# Ames Housing Prices

Brian Yi, Alexa Edwards, Erica Chen, Minerva Fang

## Introduction

**Purpose:** The purpose of this project is to develop and validate a model that can assist with prospective house owners in estimating house prices depending on the qualities they are looking for in their future home.

**Method of Approach:** This group project focuses on building and testing a multivariate linear regression model from the Ames Housing dataset. Our dataset is split between a training set that is used to "train" our model, and a testing set that we "test" our model on. First, we use best subsets regression, backward elimination, forward selection, and step-wise regression to build our initial model. We conduct some individual t-tests for slope to measure the significance of the predictors in our model. We also exmaine the variance inflation factor (VIF) to detect any multicollinearity between our predictors. Next, we do some residual analysis through residual vs fits plots, histogram distributions, and normal quantile plots. We also compute standarized residuals to pinpoint potential outliers.

Our initial model has some insignificant variables so we transform these predictors so that they have a stronger correlation with our response variable, `Price`. With this transformed model, we conduct the same hypothesis tests and residual analysis that we did for our initial model. Finally, we cross-validate our transformed model with the testing set to evaluate its effectiveness in predicting house prices. To finish our report, we take our final model out for a spin by predicting a particular housing price to see if the results are reasonable.

**Results:** Our final transformed model fits our data well and fulfills all the metrics that we evaluated it with. There are a couple of things about how our methodology can be potentially lacking that will be addressed thoroughly in the conclusion.
