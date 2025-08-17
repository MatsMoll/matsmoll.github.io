---
layout: post
title: "Smoke Testing for ML Pipelines"
categories: [ML Testing, CI/CD]
permalink: /posts/smoke-testing-ml-pipelines
---

When working on machine learning pipelines, most of the bugs you’ll encounter won’t be “the model isn’t accurate enough.”
They’ll be things like:

- A preprocessing step crashing because a column is missing.
- Data arriving in a slightly different format than your code expects.
- A key feature is accidentally dropped during processing, and the model trains without it

These problems aren’t glamorous, but they’re the ones that break production the fastest.
The fix? Add smoke tests that run the pipeline end-to-end with tiny, synthetic datasets.

## Why small-scale smoke tests work
The idea is simple: run your pipeline from start to finish using data that’s easy to generate and doesn’t require big hardware. The goal isn’t to prove the model is good — it’s to prove the pipeline still runs and still respects its expected input and output formats.

With this approach, you can catch schema changes, broken preprocessing logic, or missing dependencies before you commit to running a full training job.

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

## Defining only a few columns with RandomDataSource
Let’s make a simple rule:
- Wines with alcohol > 12 get quality = 8
- Wines with alcohol <= 12 get quality = 4

We’ll specify just those two columns, and let the rest be random within their contract bounds:

```python
from aligned.source import RandomDataSource

patterned_source = RandomDataSource.with_values({
    "alcohol": [11, 13, 15, 9, 14],
    "quality": [4, 8, 8, 4, 8]
})

store = await ContractStore.from_dir(".")
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


## Generating prediction samples
Now that we have a model, let's make some predictions to test and validate that the model picked up on the known pattern.

Thakfully we can generate new prediction samples by defining only the features that matter for your pattern — here, alcohol — and letting everything else be random:

```python
pred_df = await store.contract(WineQuality).select(X.columns).features_for({
    "alcohol": [11, 14]
}).to_polars()

preds = model.predict(pred_df)

assert preds.to_list() == [4, 8]
```

If your preprocessing and training logic are correct, the model should reproduce the pattern.

## Conclusion

Why this works well for smoke tests
This style of testing gives you both:

End-to-end execution — all features are present, pipeline runs without crashing.

Behavioural verification — you know the correct outputs for certain inputs, so you can test that the model’s logic is intact.

It’s a lightweight but powerful check you can run before heavy training, making it perfect for CI/CD.
