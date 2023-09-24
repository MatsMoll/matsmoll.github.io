---
layout: post
title: "DX can be more than just 'It Feels Good'"
categories: [DX, UX, Software Design]
permalink: /posts/dx-is-more-than-it-feels-good
---

# Evaluating AI Models in Production

Knowing how well an AI model is performing is essential. Therefore, a common practice is to have a dedicated data set used for evaluation when training our model.

Even though such an evaluation process is valuable and often finds out if the model is good enough, it may still have flaws.

The main issue is that it may not actually represent accurate performance metrics. Furthermore, there are an incredible amount of different data errors that could happen. E.g., leaking information either through features or time. 
Therefore, inaccurate performance metrics can quickly happen if we have such data errors, making the evaluation reports misleading or invalid. 
As a result, the most accurate way to monitor a model is through production.

However, evaluating the model performance in production can often be down-prioritized or an afterthought. Potentially ending up in a situation where we never add evaluation in production.

However, evaluating models in production is simple in theory.

## What is needed?
There are two things needed to measure performance in production.

1. The predictions
2. The ground truths

We will already have the ground truth stored somewhere for most classification and regression models, as it is likely that we have trained a model ourselves using some version of a decision tree or regression model. 

Furthermore, we hopefully have the predictions stored somewhere as we will need to serve them to the end user. However, the predictions and ground truths can be stored in different locations or data sources if using a microservice architecture, making it a bit more tricky.

One solution will be to ingest the predictions in a data warehouse or data lake. However, we must still set up data loaders and create processes that combine the data.

Another solution would be to use something like gantry.io that generates evaluation reports for you. However, this means you must send your data to a third-party service, which is not always possible or acceptable.

Therefore, I wanted to create a solution that enables easy evaluation of AI models in production using the business's existing infrastructure.

## Taxi ETA model
Let's look at an example of a Taxi ETA model and how we could evaluate performance in production.

Imagine we have a PostgreSQL table that contains a list of trips taken, where they started, where they ended, and the duration of the trip. Therefore, we can use the duration as the ground truth.

Furthermore, we store the predictions in another table containing the predicted duration, a timestamp of when it was predicted, and the model used. Therefore, we now have the location of our predictions as well.

But how should we model this?

I will use `aligned` for this, so let's start modeling the ground truth.

```python
from aligned import PostgreSQLConfig, UUID, Int32

db = PostgreSQLConfig(env_var="PSQL_URL")

@feature_view(
	name="trip",
	batch_source=db.table("trips")
)
class Trip:
	trip_id = UUID().as_entity()

	duration = Int32()
	
	number_of_passengers = Int32()
```

The code above defines the schema we have stored in the Postgres database named `trips`. At the same time, we have specified that we will select the features using `trip_id` as the "identifier" or entity. 

However, we have not yet defined our `duration` column as the ground truth. Mapping the ground truth is done in our model definition.

```python
from aligned import model_contract, Float, UUID, EventTimestamp
from trips import Trip, db

trip = Trip()

@model_contract(
	name="trip_duration",
	features=[trip.number_of_passengers, ...],
	predictions_source=db.table("trip_predictions")
)
class PredictedTrips:
	trip_id = UUID().as_entity()
	
	predicted_duration = trip.duration.as_regression_label()
```

Again, the above code defines a model contract where we list the features that we use as input to the model, the table used to store the predictions in, that we will query the predictions using the `trip_id`, and that the `trip.duration` column is the label that we predict.

## Evaluate the model
The above code has everything that we need. We have defined where we store our ground truths, predictions and that the `duration`  column is the target for our regression model. However, now we need to evaluate the predictions.

```python
from sklearn.metrics import mean_squared_error
from aligned import FeatureStore, PostgreSQLConfig

store = FeatureStore.from_dir(".")

db = PostgreSQLConfig("PSQL_URL")

entities = db.fetch("""
	SELECT trip_id
	FROM finished_trips_view
""")

eval = await store.model("trip_duration")\
	.with_labels()\
	.predictions_for(entities)\
	.to_pandas()

mse = mean_square_error(eval.predictions, eval.ground_truths)
```

In the above example, will we select the instances where a ground truth exists, join the predictions for them, and the ground truths themselves.

Our response from the database contains all of the data in one data frame but nicely separates the different information for us. They make it easy to compute metrics using the attributes of `ground_truths` and `predictions`.

## Conclusion
Evaluation is an essential part of AI applications. However, evaluating how well our models perform in production can often be down-prioritised, even though this is the most accurate metric for our model, as data leakage is way less likely in such an environment.

Furthermore, using the `aligned` package makes it easy to define the relationship between our ground truths and predictions. While also defining where we to fetch the different information. Such a setup makes it easier to load the data needed to evaluate our models living in production.
