---
layout: post
title: "Smoke Testing for ML Pipelines"
categories: [ML Testing, CI/CD]
permalink: /posts/smoke-testing-ml-pipelines
---

When working on machine learning pipelines, most bugs aren’t about bad models — they’re about broken plumbing.

They’ll be things like:

- A preprocessing step crashing because a column is missing.
- Data arriving in a slightly different format than your code expects.
- The input schema for the model doesn’t match the data provided at inference.

These problems aren’t glamorous, but they’re the ones that break production the fastest.
The fix? Add smoke tests that run the pipeline end-to-end with tiny, synthetic datasets.


## Why small-scale smoke tests work
The idea is simple: run your pipeline end-to-end using synthetic data that doesn’t need big hardware. The goal isn’t to prove the model is good — it’s to prove the pipeline still runs and still respects its expected input and output formats.

With this approach, you can catch schema changes, broken preprocessing logic, or missing dependencies before you commit to running a full training job.

A smoke test saves you from an 8-hour training run that crashes on a missing column. Rather it would catch it in seconds. 

So how do we create these datasets?

## Generating synthetic data
You can create synthetic test data in two main ways:

- Fully randomised data — great for checking that schemas match and code runs, without caring about meaning.
- Partially controlled data — lets you embed known patterns so you can confirm the model can still detect them.

If you’re working with a framework that defines your data contracts or feature views, you can often generate random data directly from those definitions. Here is an example using [aligned](https://github.com/MatsMoll/aligned) and the wine quality dataset:

```python
@data_contract()
class WineQuality:
    fixed_acidity = Float64()
    volatile_acidity = Float64()
    citric_acid = Float64()
    residual_sugar = Float64()
    chlorides = Float64()
    free_sulfur_dioxide = Float64()
    total_sulfur_dioxide = Float64()
    density = Float64()
    pH = Float64().bounded_between(0, 14)                  
    sulphates = Float64()
    alcohol = Float64().bounded_between(0, 20)
           
    quality = Int64().bounded_between(0, 10)
    
df = await WineQuality.n_examples(100).to_polars()
```

This contract enforces valid value ranges, or infer defaults based on the data types — which means we can safely auto-fill any column we don’t explicitly set.

Random data is good, but sometimes you need controlled patterns.

## Defining a known pattern
Let’s make a known pattern with the following rules:
- Wines with alcohol > 12 get quality = 8
- Wines with alcohol <= 12 get quality = 4

We’ll specify just those two columns, and let the rest be random within their contract bounds:

```python
from aligned.source import RandomDataSource

# Defining the pattern
patterned_source = RandomDataSource.with_values({
    "alcohol": [11, 13, 15, 9, 14],
    "quality": [4, 8, 8, 4, 8]
})

store = await ContractStore.from_dir(".")

# Switching to the new data source
store = store.update_source_for(
    WineQuality,
    patterned_source
)
```
Because the other columns in WineQuality aren’t defined here, they’ll be filled in automatically with valid random values. However, all rows will have the same random value, creating a strong deterministic relationship between the alcohol and quality columns. With that setup, we can now load the dataset and train a model.

```python
from sklearn.ensemble import RandomForestClassifier

df = await store.contract(WineQuality).all().to_pandas()

X = df.drop("quality", axis=1)
y = df["quality"]

model = RandomForestClassifier().fit(X, y)
```
Next, let’s validate the pipeline by predicting on a controlled test-set.

## Generating prediction samples
Now that we have a model, let's make some predictions to test and validate that the model picked up on the known pattern.

Thankfully we can generate new prediction samples by defining only the features that matter for your pattern — here, alcohol — and letting everything else be random:

```python
pred_df = await store.contract(WineQuality).select(X.columns).features_for({
    "alcohol": [11, 14]
}).to_polars()

preds = model.predict(pred_df)

assert preds.to_list() == [4, 8]
```

With this, we can finally check that our pipeline loads the correct data and trains a model that finds the known pattern.
Furthermore, if you connect a proper model registry to this pipeline could you also test that those integrations work as expected.

## Conclusion
Smoke tests won’t guarantee that your model is cutting-edge, but they will guarantee that your pipeline is alive and dependable. By mixing randomized data for schema validation with controlled patterns for behavioral checks, you catch the kinds of issues that cause real outages. Missing columns, broken preprocessing, or silent feature drops.

You can even extend these tests beyond data and training by wiring in a model registry. Verifying that a trained model can be stored, retrieved, and reloaded successfully gives you end-to-end confidence — from raw data all the way to deployable artifacts.

These checks are fast, cheap, and easy to automate. Making them perfect for CI / CD. Enabling us to check every code change in seconds, long before you commit resources to a full training run. The result is fewer surprises, faster iteration, and more confidence that your pipeline is doing what you expect. So when it’s time to improve accuracy, you can focus on the model, not firefighting infrastructure.

For more details on using aligned, check out the [project on GitHub](https://github.com/MatsMoll/aligned) or its [documentation](https://www.aligned.codes/).
