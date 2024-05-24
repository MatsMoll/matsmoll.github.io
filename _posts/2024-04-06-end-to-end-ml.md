---
layout: post
title: "How I created an end-to-end ML platform for a personal banking AI"
categories: [MLOps, AI Products]
permalink: /posts/end-to-end-banking-ai
---

This Christmas I wanted to set my-self a fun challenge. I wanted to create an end-to-end AI product, with model serving, with performance monitoring, with a data catalog, with propper data engineering, with orchestration, with an iOS front-end, and running only on local compute. Not to much to ask, right?

## The ML Problem
However, to implement such a project was an ML related problem needed. Therefore, I thought it would be fun to create an ML model that predicted my expenses for the next month, but grouped by categories like transportation, groceries, sports, etc.

Potentially not the most useful model, as I know fairly well how my expenses will be because of a fairly good banking app. However, since I know the data well, will it be easier to figure out when the model is way off, which makes it the perfect use-case for an end-to-end ML platform.

However I started looking into the API of my bank, and saw that I was lacking the data I needed...
More precisely, I was lacking some of the data that told me if it was spent on transactions, groceries, etc. 
However, I thankfully had it for all my in-person VISA card transactions.

Therefore, this post will go info how I created a model that classified my bank transactions, while ensuring high quality datasets, how I made it easy to maintain models that depend on each other, and how I monitored the performance in production.

## The Source Data
Before I can start doing anything really practical, would I need to get my hand on some real data. Or at least understand how the data would behave to some degree.

This lead to write a simple API integration with my bank that loaded my transaction data given a time period. E.g. from 1 January 2023 to 1 January 2024, or by providing a number of records to load.

I also defined the schema of the response I expected, which you can see bellow. We will come back to the package used to define the schema, as it is not your ordinary `pydantic` model, and for a very good reason.

```python
class Transaction:
    transaction_id = String().as_entity()
    
    amount = Float()
    text = String()
    
    user_id = String().description(
        "Only a hard coded value, as it is only my own user"
    )
    account_id = String()
    
    accounting_date = Timestamp()
    interest_date = EventTimestamp()


    transaction_type = String()

    card_details = Json().is_optional().description(
        "Is only set where transaction_type = VISA VARE"
        "Contains a lot of interesting details tho"
        "E.g. merchant category, location, etc."
    )
```

However, rather then going for an ordinary SQL database, did I go for more of a data lake solution. Therefore, all of the data would be loaded from an API, but stored in a Parquet file locally.
So how would I manage the data? This is where `aligned` comes into play. 
Aligned is a package used to manage data for ML applications, and is also the package used to define the schema for my transaction data.

However, in addition to defining schemas, can we add different types of sources. Like a load source, materialised sources, and stream sources. But also define how often we expect them to be updated. Which can be done with the following:

```python 
@feature_view(
    name="transactions",
    source=api_source,
    materialized_source=FileSource.parquet_at(
        "source_data/transactions.parquet"
    ),
    acceptable_freshness=timedelta(days=3),
    unacceptable_freshness=timedelta(days=6),
)
class Transaction:
    transaction_id = String().as_entity()
    amount = Float()
    ...
```

### ML Data Management
Now that I finally had the source data loaded was it time to create the model. 

This ment a few things:
- I needed to transform the raw data to extract the merchant category out from from the json data.
- I needed to create a dataset where the merchant category is the ground truth.
- I needed to define which inputs to use for the model.

Thankfully was the first step very simple. It was done by adding a new property to the schema, but that used the `card_details` as the source. Therefore, creating a "type-safe" computation, with data lineage and a lot of goodies.

```python
@feature_view(...)
class Transaction:
    transaction_id = String().as_entity()
    ...
    
    card_details = Json().is_optional()
    merchant_name = card_details.field(
        "merchantName", 
        as_type=String()
    )
```


When it comes to the second part about defining the ground truth do `aligned` have an elegant method. `aligned` introduces the concept of model contracts which tell us what a model is predicting, their ground truth, and a lot of other related metadata. As a result can we use a similar syntax as the transformation to define that the `merchant_name` should be the ground truth and the predicted value.

```python
transaction = Transaction()

@model_contract(
    name="transaction_category",
    input_features=[...],
)
class TransactionCategory:
    transaction_id = String().as_entity()
    predicted_at = EventTimestamp()
    
    predicted_category = (transaction.merchant_name
        .as_classification_label()
    )
```

By adding the `as_classification_label` will we tell `aligned` where to find the ground truth, what we are predicting, and what the model should name the prediction, as it is set to the `predicted_category` field.

### Create a Training Dataset

In order to train a model would I need a dataset, and thanks to the defined relationship in the last section will it be easy to create. The only thing I need to define is who I want to create a training dataset for, and in my case do I want it for all transactions where we have a ground truth value. However, the ground truth check will be done automatically, so all we need to care about will be for who to load our data for. Therefore, this can be loaded with the following.

```python
entities = Transactions.query().all_columns(limit=None)
```

And to create a dataset can we do the following.


```python
store = await FeatureStore.from_dir(".")

datasets = (store.model("transaction_category")
    .with_labels()
    .features_for(entities)
    .train_test_validate(
        train_size=0.7,
        validate_size=0.15
    )
)
```

This will make sure of a few things.
- We have a ground truth in each row.
- We do not train on data that was created after we would have made the prediction. Also known as point-in-time correct data.
- Making sure we do not have data leakage in time, by splitting the dataset based on event timestamps. If there exist no will it do a random split.

From here we can load different datasets by accessing the `train`, `test`, or `validate` property.

```python
train_data = await datasets.train.to_polars()

model = LogicticRegression()
model.train(
    X=train_data.input, 
    y=train_data.labels
)

for dataset in [datasets.test, datasets.validate]:
    eval_data = dataset.to_polars()
    preds = model.predict(eval_data.input)
    
    accuracy = (preds == eval_data.labels).mean()
```

And that's it, I finally had my model which I then stored in a MLFlow registry. I skipped the MLFlow implementation, to keep it simple.

### Use the model
There are a wide range of ways to deploy a model. A common way is use a MLFlow server, which can either be started locally or a data platform such as Databricks.

This is where aligned helps even more, as it makes it possible to describe where, and how to use a model.
Therefore, we can define that we have an MLFlow model, served at a specific url. Or for simplicity sake, since the model has a low memory footprint can I use it in memory. Therefore, I modified the model contract to the following

```python
@model_contract(
    name="transaction_category",
    input_features=[...],
    exposed_model=ExposedModel.in_memory_mlflow(
        model_name="transaction_category",
        model_alias="Champion",

        prediction_column="predicted_category",
        predicted_at_column="predicted_at",
    ),
    output_source=FileSource.parquet_at(
        "transaction_category_preds.parquet"
    )
)
class TransactionCategory:
    transaction_id = String().as_entity()
    predicted_at = EventTimestamp()
    
    predicted_category = (transaction.merchant_name
        .as_classification_label()
    )
```

Therefore, enabling me to just define who to predict for, and the feature loading, interacting with a model in memory or through an API will be done for you, metadata will be added automatically.

```python
predictions = await store.model("transaction_category").predict_over({
    "transaction_id": [...]
}).to_polars()
```

Or if I want to store them in the `output_source`, I could do it with the following.

```python
await store.model("transaction_category").predict_over({
    "transaction_id": [...]
}).upsert_into_output_source()
```

### The model input

Until now have I skipped one crucial part. I have not talked about the input I used to the classification model.

Since I know the transaction text description would probably be the best feature, did I land on using an embedding model, and use the embedding vector as the inputs. 

Thankfully, `alinged` have an easy integration to setup a complete model contract with Ollama integration.

```python
TransactionEmbedding = ollama_embedding_contract(
    contract_name="transaction_embedding",
    text=transaction.text,
    endpoint="http://host.docker.internal:11434",
    model="mistral:latest",
    entities=transaction.transaction_id,
    output_source=FileSource.parquet_at(
        "transaction_embedding.parquet"
    )
)
```

From here the input can be sent into the original transaction model.

```python
transaction_embedding = TransactionEmbedding()
@model_contract(
    name="transaction_category",
    input_features=[transaction_embedding.embedding],
    ...
)
class TransactionCategory:
    transaction_id = String().as_entity()
    predicted_at = EventTimestamp()
    
    predicted_category = (transaction.merchant_name
        .as_classification_label()
    )
```

This will make it possible for `aligned` to ensure that we never break the transaction category model, by making breaking changes to the embedding model.

### Evaluate the model

Finally, I had a model to use, and I finally created an API and an iOS widget to send a request to predict over my new transactions each day. This ment I accumulated predictions all the time, and after some time I could finally evaluate it.

Thankfully, `aligned` knows where we store predictions, and where the ground truths are, and knows that it is a classification problem. Thereby making evaluation super easy.

Thereby, we can open the aligned UI and get confusion matrix, accuracy, precision, and recall metrics without doing anything.

Furthermore, we can add a model version variable to the model contract in order to compare different models against each other.

```python
@model_contract(...)
class TransactionCategory:
    transaction_id = String().as_entity()
    predicted_at = EventTimestamp()
    model_version = String().as_model_version()
    
    predicted_category = (transaction.merchant_name
        .as_classification_label()
    )
```

## Conclusion

Building an end-to-end ML project have made it clearer the wide range of problems that ML offers. Furthermore, `aligned` makes it possible to get more done with less work, by moving the focus away from model first. And focusing on how the ML data flow will behave, and then say how to integrate the model.
