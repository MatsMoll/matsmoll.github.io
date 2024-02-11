---
layout: post
title: "The DBT of AI"
categories: [Data Managment, AI Products]
permalink: /posts/the-dbt-of-ai
---

AI is evolving rapidly with new models every day. However, it is easy to get lost in the hype of new models and forget how AI will be integrated at a system-wide level.

I missed such system-wide tooling, so I created [`aligned`](https://github.com/MatsMoll/aligned) to make it easier to understand, develop, and evaluate AI and ML data products.

## My Annoyances
Before we delve into my goals with `aligned`, it is important to understand what I felt was missing in the current AI stack.

### Code Completion
One of the most surprising things about moving to the AI ecosystem was the lack of code completion and tooling to prevent bugs. This was especially clear when coming from a strictly typed language like Swift, where types need to be enforced, and the use of raw strings is seen as a code smell.

In contrast, the AI landscape was the clear opposite. Almost no types, and the types that were present were not enforced either. But also, almost all code was described using raw strings. For example, how `pandas` do transformations - `df["new"] = df["a"] + df["b"]`.

Why is this raw string usage so bad? It makes it almost impossible for linters to find semantic errors, like mistyped attributes or invalid datatype operations. But it also makes it very hard to provide code completion, and help the developer understand what is possible to do. Therefore, reducing the dependence on package tutorials and documentation.

### Implicit Schemas
Continuing on the usage of strings, I often found that Python programs heavily relied on implicit schemas.

One such example would be `yaml` config files, and how they often get accessed through dictionaries.
Below is a fairly common training pipeline that I found in a Medium post.

```python
config = load_config("my_config.yaml")

# load data
data = pd.read_csv(
    os.path.join(
        config["data_directory"], 
        config["data_name"]
    )
)

# drop id column
data = data.drop(config["drop_columns"], axis=1)

# Define X (independent variables) and y (target variable)
X = np.array(data.drop(config["target_name"], 1))
y = np.array(data[config["target_name"]])

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config["test_size"], random_state=42
)

# call our classifier and fit it to our data
classifier = KNeighborsClassifier(
    n_neighbors=config["n_neighbors"],
    weights=config["weights"],
    n_jobs=config["n_jobs"],
)
```

If any of these keys are missing, or potentially in the incorrect data type, this pipeline can fail. For example, if an integer is loaded as a string.
Rather, I want these implicit schemas to be defined explicitly, making it possible to fail faster and make it clearer which information is available.

### Implicit Logic
Another issue I found was that getting a Birds Eye view of our AI products was hard.
We mostly documented our AI products manually, but it was often only showing the AI product, and not how it integrated into the system on a larger scale.
Furthermore, the technical documentation was either missing or out of date.

Therefore, questions such as “what are the goals of the model,” “who owns the model,” “which data are needed to run the model,” “what are we predicting,” “where do we store predictions,” “where can I run the model,” “how often do we expect the model to run,” “where do we store training datasets,” and so much more were left unanswered.

Often I found that a lot of these questions were described implicitly in code, but outdated in the docs.

### Default Implementation
Implicit logic and implicit schemas are somewhat similar. However, defining both of them can at first seem like more of a pain than anything.
But defining the implicit schemas and logic explicitly can help derive default implementations.
Such as data validation, setting up data freshness checks, row duplication checks, model evaluation, model performance monitoring, and so much more.

Therefore, explicit schemas and logic do not only help with failing faster, but it also helps with implementing reasonable functionality in less time.

### Inflexible Data Sources
Lastly, I was surprised at how impractical it was to work with data sources in AI.
We either created custom data wrappers, got locked into using one data warehouse, or we copied the data of interest to a local file.

However, we often wanted the possibility to combine our data warehouse with experimental sources, like local files or transactional databases, but this was either very convoluted or impossible to do.

## The Goals
As a result, I wanted to improve all of this.
Therefore, I wanted to create a tool that provides a good developer experience by offering code completion and catching errors earlier.
I wanted you to explicitly define your expectations, so that we can fail faster but also go further with less work.
I wanted

 the technical documentation to be generated from code, rather than being dependent on us keeping it up to date.
And of course, I wanted to make it possible to mix and match data sources, so you can run experiments in less time.

## Aligned
This is how `aligned` was created.
Now, if you are familiar with facts and dimensions, this should hopefully not be too unfamiliar.

At its core, `aligned` will introduce two concepts: `feature_view`s - similar to a dimension, and `model_contract`s.

### Model Contract
A model contract is crucial in `aligned`. This is where we define all the metadata for our model, such as intent, input features, and output.

To make this clearer, I will present a side project I created where I wanted to categorize my bank transactions.

```python
@model_contract(
    name="transaction_category",
    features=[
        transaction.text_embedding
    ],
    prediction_source=FileSource.parquet_at("pred_data/transaction_category_test.parquet"),
    dataset_store=FileSource.json_at("datasets/transaction_category.json"),
    exposed_at_url="http://server:8000/openapi.json", # An internal docker compose URL
    acceptable_freshness=timedelta(days=2),
    unacceptable_freshness=timedelta(days=4),
)
class TransactionCategory:
    transaction_id = String().as_entity()
    
    predicted_at = EventTimestamp()
    
    model_version = String().as_model_version()

    predicted_category = (
        transaction.merchant_category
            .as_classification_label()
    )
```

So, what is happening here?
First of all, we define a model that will take an embedding as the input (`features`) and that the model will produce a `predicted_category` together with some other metadata, like `predicted_at` the `transaction_id` it predicted for, and the model used in its `model_version`.

Furthermore, we also define where our predictions are stored (`prediction_source`), where the model is exposed (`exposed_at_url`), and how often we expect it to predict (`acceptable_freshness`, `unacceptable_freshness`), and where we store the train, test, validate sets (`dataset_store`).

However, we reference a `transaction` variable in both the input features and when defining what we predict, so what is this?

### Feature View

This is where our feature views come into play. Very similar to a dimension, as we define different information associated with some kind of entity.

This is what our `transaction` variable is from the previous example.

```python
@feature_view(
    name="transactions",
    source=api_source,
    materialized_source=FileSource.parquet_at(
        "source_data/transactions.parquet"
    ),
    description="The bank transaction made by a user",
    contacts=["MatsMoll"],
    acceptable_freshness=timedelta(days=3),
    unacceptable_freshness=timedelta(days=7),
)
class Transaction:
    transaction_id = String().as_entity()

    user_id = String()
    account_id = String()
    
    accounting_date = Timestamp()
    interest_date = EventTimestamp()

    amount = Float()
    abs_amount = abs(amount)

    text = String()
    transaction_type = String()

    card_details = Json().description(
        "Is only set where transaction_type = VISA VARE"
    ).is_optional()

    is_expense = amount < 0
    is_income = amount > 0
    
    merchant_category = card_details.field(
        "merchantCategoryDescription", String()
    ).is_optional()
    
    text_embedding = text.embedding(
        EmbeddingModel.huggingface("all-MiniLM-L6-v2")
    )

transaction = Transaction()
```

There are a lot of things happening here, so let's go through it.
First of all, we define all the information we expect a transaction to have, and its associated datatype.

Furthermore, we define where all this data will be fetched from `source`, but since this example will load from an API, we also write this data to a `materialized_source` working as a cache.

And similar to our `model_contract`, we define the expected freshness of this data. Making sure we are aware if our data is older than our expectations of 3 days.

Furthermore, by default, all columns will be required, which is why you see the `card_details` have a `.is_optional()`, as it can be missing in some scenarios.

Also, notice that we can add documentation directly on a feature with `.description(...)`, in case there are some extra contexts needed to justify the setup.

But not only that, we can even provide transformations in the feature views (`is_expense = amount < 0`). And this without any strings. Therefore, providing both code completion, and linters can catch errors for you.

### Data Lineage

All of this is nice to describe, but it gets better.
You see, since we have all of

 these transformations and references, `aligned` will automatically collect data lineage through features, and even more interestingly, through models.

Meaning, if we had created a downstream model that depends on the `predicted_category`, `aligned` would know and show it to you.

Therefore, making it easier to provide a valid overview of your ML products.
<video width="100%" controls>
  <source src="/assets/videos/aligned-overall-data-lineage.mp4" type="video/mp4">
View Data Lineage
</video>

But it also makes it possible to prune unneeded transformations and debug transformations.
<video width="100%" controls>
  <source src="/assets/videos/aligned-test-transformations.mp4" type="video/mp4">
Test data transformations
</video>

Just look at how we can ask for the `question_embedding`, and it knows that you need to provide a `question` and a `description`.

## Conclusion
The AI landscape has evolved a lot in the past years, and we have a lot of good tooling to simplify the development of AI products. However, managing ML products on a system scale still needs a lot of implicit knowledge. Something that can also lead to extra work, longer onboarding times, and confusion.

As a result, I developed `aligned`, which tries to manage the data from AI products. Therefore, somewhat becoming the DBT of AI. As a result, `aligned` simplifies the development, management, and understanding of how our AI products actually behave.

So if this is interesting, try out [`aligned`](https://github.com/MatsMoll/aligned) and let me know what can be done to improve it further.
