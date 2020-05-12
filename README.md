# heart-disease-predicts
Lets try to predict whether a patient has heart disase

Started out by plotting some of the variables in our data to get a look at the distribution.

![variable_distributions](https://user-images.githubusercontent.com/23482152/74297928-504c0500-4d16-11ea-9b9d-8b454591e0a9.png)

Along with the distribution we want to see if there are any missing values in our data.


![NA_values](https://user-images.githubusercontent.com/23482152/74298527-10861d00-4d18-11ea-8fbc-c51913978087.png)

Luckily we have a clean dataset that is ready for analysis.

Decided to go with a Logistic Regression.  Due to the small size of our dataset this seems to be the most efficient and best option.

Once we run our model we want to check out how it scores on both our train subset and our test subset.

![Model_scores](https://user-images.githubusercontent.com/23482152/74298204-0879ad80-4d17-11ea-86da-f23f95f9c483.png)

Once we have our scores, it could always be better, but in the meantime prior to making our model better we can check out some of our metrics.


![classification_report](https://user-images.githubusercontent.com/23482152/74298282-3fe85a00-4d17-11ea-99a1-dafa03785231.png)

Lastly we have our matrix that shows how our predicted values compares to our actual values.

![actual_vs_predicted](https://user-images.githubusercontent.com/23482152/74298318-5abace80-4d17-11ea-814f-1f4223e729e7.png)
