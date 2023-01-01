---
layout: post
title: "How I created the most elegant MLOps tool"
permalink: /2022/12/31/How-I-created-the-most-elegant-MLOps-tool.html
categories: [MLOps]
---

I wanted to write about this topic for a while but struggled to frame it correctly.  And finally, I landed on just telling the origin story, with some of its invisible sub-stories. So grab a coffee, and let's start on how I created the most elegant MLOps tool.

## The beginning

Before we dive into machine learning-related topics because the final MLOps tool has its roots in a domain that may seem obscure, however, it has been one of the most impactful and essential experiences yet.

### Not so unrelated domain after all?

Even though I ended up in the machine learning domain, I started developing iOS applications around ten years ago. I bought an iPod touch as a child and soon wanted to create applications. This interest continued for a while, and I got a job as an iOS developer after roughly four years of hobby development before graduating from high school.

The experience is necessary to mention, as it educated me on the importance of UX in software development. And I often find myself talking to developers that think UX is all about the graphics or only for frontend developers. But I'm afraid I have to disagree with this statement. I find programming principles like DRY, SOLID, and CLEAN code to describe UX principles in an opinionated and use-cases-specific way. This background led to a slight shock when starting to develop machine learning systems as I found the existing tooling to not focus on UX the same way as frontend tooling did. I want to write a post that dives more into the details here, but for this post. It was vital for me to focus on improving error prevention, improved contextual API -awareness, and -discoverability, and struck a nice balance between simplicity and flexibility.

### The beginning of my machine learning role

One summer, I managed to get an internship at Otovo, one of Europe's biggest solar companies. They needed a way to detect solar panels that did not work as expected and wanted to see how we could use machine learning for the problem. I ended up working as the machine learning engineer in our cross-functional team, as I managed the data, developed the backend, and helped with which models to use. Therefore, I converted the data scientists' work into code used in production, where problems started to arise.

#### Data collection
First of all, data collection could have been faster. The data scientist needed to ask me for either new data or a new feature to analyze. It led to slow development, as they needed to wait for my crafting the SQL queries, finding the correct tables, setting the proper constraints, combining multiple sources, and then creating the dataset in whatever file format made sense. However, it also led to me using time on what felt like low-value work, as given the correct tools, the data scientist would be able to do it themself.

#### Quality and consistency
Furthermore, the resulting products that the data scientist produced could have been better quality. And to be fair, I do not blame this on the data scientists. They are very well educated and know their stuff. However, the tooling needed to provide a guarantee of correctness or quality. For instance, I remember each data scientist had one notebook they worked in and therefore had their variations of pre-processing function. This setup made sense as they wanted to test different features and ideas. However, they still had some standard features, and the results needed to be reproducible when presented. Because some of the features that were supposed to be shared were slightly different, I noticed this because we had multiple data scientists, as it took a lot of attention to find minor bugs when skimming through the code. Such inconsistencies made my work as an ML engineer harder, as I needed to decide which DS's work to use or fix the logic myself.

#### Takeways
At the end of the project, it was clear that data management and consistency were on of the main challenges. As a result, we took down the production database a few times because we prioritized fast development over a highly efficient architecture. But the product was up and running and working as expected.

## A steep learning curve
After the internship, I got an offer of a contract to work for Otovo, and I said yes because I enjoyed the people and challenges so much.
This choice led to improving the existing system and fixing needed problems. But it also meant starting on new ML projects. Therefore, presenting more demanding challenges to solve.

### Started at the bottom
The next task was to create a system to make our sales teams more efficient. We wanted to create a ranking system for our customers based on how likely they were to buy solar power. First, however, we needed a real-time system since the sales team called the customers as fast as possible. At the same time, the team got reduced from a team of five to one data scientist and me.

## The creation of our first tool
I did not want to make the same errors and handle the same problems as the previous project. So, I started creating a new library where we could define features as composed components while simplifying the data loading. Having one interface for SQL, files, or APIs makes it easier to define pre-processing, splitting data sets, easily accessible EDA, model training, metrics evaluation, and finally, storing models. All while making the API discoverable and contextual. For instance, getting code completion for model evaluation was only possible with training or loading a model first. The features of such a library had a lot of potentials, and it fixed the problems it set out to solve. Therefore, our pipelines were easier to maintain, as it was apparent which part to edit. It was easier to test, simplified standardizing of features, and was faster to spin up some essential pipelines. The following code presents a small pipeline using our framework Bender. For more details, view [GitHub](https://github.com/otovo/bender). 

```python
await (DataImporters.data_set(DataSets.IRIS)
    .process([
        Transformations.exp_shift('petal length (cm)', output='pl exp'),
        Transformations.exp_shift('petal width (cm)', output='pw exp'),
    ])
    .explore([
        Explorers.histogram(target='target'),
        Explorers.correlation(input_features),
        Explorers.pair_plot('target'),
    ])
    .split(SplitStrategies.random(ratio=0.7))
    .train(Trainers.kneighbours(), input_features=input_features, target_feature="target")
    .evaluate([
        Evaluators.confusion_matrix()
    ])
    .metric(Metrics.log_loss())
    .run())
```

However, we still had some pain points, so let's walk through a few.

### An opinionated system
Providing features as a set of components was one of our goals. However, the solution in Bender changed how data scientists work with data quite drastically. Setting up transformation logic in the provided method was unnatural and required learning a new API. But it took a lot of work to convince the data scientist that the gain was worth the learning investment. Furthermore, the tooling tried to do everything, also known as end-to-end. But this is hard to do, as there are so many details and methods that data scientists want to test out. Bender tried to make this possible by defining interfaces, making it extensible for new techniques, but it was still locked into Pandas and tried to do a bit too much all at once.

### Training-serving skew
As every project behaves, is the development process an iterative process. Such a process led to a problem when we needed to update our input features. Therefore, we needed to update the training pipeline and inference pipeline. Having two pipelines to update was very maintainable, but we had no tool to ensure we aligned on the same features in different environments. Something that can lead to silent errors as we produce invalid predictions. Also known as the training-serving skew. The training-serving skew got amplified more when we wanted to explain our predictions, as we needed to reproduce the pipeline a third time. Therefore, we started looking for new solutions to our latest problems.

## Looking for open-source solutions
We mainly wanted to fix our training-serving skew and quickly realized that a single source of truth could simplify everything. An SSOT is a widespread practice when developing applications. Either by having one view model that handles the application state or as a data warehouse which is the source of truth for all data analytics. In the sense of UX principles, this recedes the possibility of errors while simplifying the user experience. Thankfully, the concept of a feature store already existed, which tries to set up an SSOT for real-time ML systems.

### The feature store
We started investigating what a feature store could do and if it would fit our needs. And it seemed to fit at a high level, as it sets up a single source of truth. Furthermore, it unified the data between batch- and inference processing. We mainly considered three different technologies, Tecton, Feast, and Hopsworks. So why does the story continue when we have these technologies?

#### Hopsworks
One of the first solutions we found was Hopsworks, which looked very powerful. However, the impression we got was that it relied on Spark, which a small team like us did not make sense. Furthermore, my old iOS developer mindset of API design wanted more flexibility and simplicity. It felt like too much boilerplate code based on what was wanted. For instance, the following code tries to create a training, test, and validation set on a feature view.

```python
try:
    feature_view = fs.get_feature_view(name = 'churn_feature_view', version = 1)
except:
    feature_view = fs.create_feature_view(
        name = 'churn_feature_view',
        version = 1,
        labels=["churn"],
        transformation_functions=transformation_functions,
        query=ds_query,
    )
    
td_version, td_job = feature_view.create_train_validation_test_split(
    description = 'churn_training_dataset_random_splitted',
    data_format = 'csv',
    validation_size = 0.2,
    test_size = 0.1,
    write_options = {'wait_for_job': True},
    coalesce = True,
)

X_train, X_val, X_test, y_train, y_val, y_test = feature_view.get_train_validation_test_split(
    training_dataset_version = 1
)
```

First of all, this is a good codebase. However, we can improve the API, and we will present some improvements later on.

#### Tecton
Not open-source, per se. Therefore, we started looking at Tecton, a SaaS solution. We found Tecton to have a more declarative API that we liked, as they focus more on which result is wanted rather than how to achieve the result. However, Tecton is heavily relying on Spark. As mentioned, this was not an option with a small team like ours.

Furthermore, there were still improvements to be made in the API. Most notably, we describe all features as SQL queries, but my iOS developer experience has taught me this. Also, using raw strings is usually a sign of future trouble. And I found Tecton to rely too heavily on strings. Just view the following code.

```python
@stream_feature_view(
    source=FilteredSource(source=transactions_stream),
    entities=[user],
    mode="spark_sql",
    aggregations=[
        Aggregation(column="amount",function="mean",time_window=timedelta(minutes=5)),
        Aggregation(column="amount",function="mean",time_window=timedelta(hours=1))
    ],
    stream_processing_mode=StreamProcessingMode.CONTINUOUS,
    batch_schedule=timedelta(days=1),
    online=True,
    offline=True,
    feature_start_time=datetime(2020, 10, 10),
)
def user_transaction_amount_averages(transactions_stream):
    return f"""
        SELECT
            user_id,
            amount,
            timestamp
        FROM
            {transactions_stream}
        """
```

Using strings in the aggregation column describes the SQL queries and the mode, and then we need to fill in the `FROM` section. We were not that satisfied, leading to the last open-source tech stack we considered.

#### Feast
Lastly, and the most used open-source solution, is Feast. However, Feast needed a Spark cluster to transform features before storing them again. Furthermore, the feature definition seemed like boilerplate code. The use of YAML for defining batch sources was unnatural to me, and it also locked us into using one source. Finally, writing YAML has UX problems, as there is no code completion, error prevention, or help when typing them.

```python
driver = Entity(name="driver", join_keys=["driver_id"])

driver_stats_source = FileSource(
    name="driver_hourly_stats_source",
    path="%PARQUET_PATH%",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)
​
driver_stats_fv = FeatureView(
    name="driver_hourly_stats",
    entities=[driver],
    ttl=timedelta(days=1),
    schema=[
        Field(name="conv_rate", dtype=Float32),
        Field(name="acc_rate", dtype=Float32),
        Field(name="avg_daily_trips", dtype=Int64),
    ],
    online=True,
    source=driver_stats_source,
    tags={"team": "driver_performance"},
)
```

All this reduced our development time, and we felt locked into a structure we did not want.

However, I knew that creating a new framework and yet another MLOps tool would take time and a lot of effort, so we looked into how we could contribute to Feast instead.

### Contributing to Feast
We decided that Feast was the best bet with the community and the existing features set. However, we needed to add transformation logic to make sense. In other words, Feast was a single source of truth for data, but we needed a single source of truth for data relation, something Feast was not.

However, Feast could fit well if we could transform features before storing them. So we started setting up a fork that could contribute to Feast. We knew it was possible, as Feast had a similar feature but only for features transformed after storage, aka on-demand features.

But again, there were a lot of problems. For example, Feast did not know that it needed to load features A and B if I wanted to combine them into a new feature C. Not knowing this led to a poor user experience, and trying to add support was challenging because of unstructured API calls and interfaces. This challenge and the combination of all other frustrations of an improved API, YAML, etc., led to the realization that it was easier to create what I wanted myself.

## The new start of my vision
After this decision, I started creating a new API in my spare time. This tool wanted to fix the single source of truth properly. Therefore, it was supposed to be a logic management tool rather than a data management tool. But I needed some clear objectives before I started.

### Simplicity to the next level
One of the most important objectives was to create an API that was so easy to use. Hopefully so simple that people would need help understanding the problem in the first place. We were trying to remove all the crazy schema definition code, manual imperative structures, etc. I also wanted to remove all the need for this Pandas-like syntax where we repeat the source continuously `df["c"] = df["a"] + df["b"]`. Instead, I wanted something like SQL, which is closer to how we think `c = a + b`. Such an API would also remove the need for all our strings, which leads to the next objective.

### Error prevention
One of Nilsen's heuristics from UX theory presents that good UX is a system that prevents errors from happening in the first place. Removing strings is something that helps with this point. For instance, a compiler can say to the data scientist that `c = a + b` is valid, but it is impossible if we have `df["c"] = df["a"] + df["b"]`. Therefore, an API that relies more on compilers and the type system will help data scientists write higher-quality projects, but this needs to be more leveraged in libraries.

### Flexibility
Learning from earlier projects, making the framework flexible and adaptable was essential to fit a wide range of needs. Using any data source, testing multiple sources together, or swapping out some parts of the system was necessary. But there was still a clear objective needed for each component. The tooling, therefore, required clear interfaces that others could adopt but also easy to extend in the future.

### Scalability
Making the tool able to scale a company's growth was necessary to make it relevant. Therefore, I could not expect that Pandas was the tool to use, but I still wanted pandas and low-cost solutions to be supported. However, I wanted to make it possible to switch from Pandas to Polars, to Dask, to Spark with as few lines as possible. Furthermore, the same logic needed to be available in stream processing, so data sources like Kafka and the lesser-known Redis streams required support. Therefore supporting multiple processing engines batch- and stream sources were essential for a scalable solution.

### Safety
Lastly, I wanted to focus on safety as I have noticed that many libraries rely on dill and pickle. I avoided these libraries by design as they are not supposed to be a safe solution. So even though this is a fast and easy way of solving the problem, it did feel like a proper way of doing things. But we still needed a way of providing a single source of truth for transformation logic, and this decision has shown to lead to some exciting advantages.

I could go on about objectives, but this sets a good foundation.

## The new solution
I needed to clarify a lot of use cases in the beginning. However, one thing was clear as gold, and it was the API design.

### The API
As mentioned, I wanted to be as close as possible to SQL. In other words, I wanted no need to repeat the source data frame `c = a + b`. But I also wanted it to be clear if this operation was allowed. For example, a string operation should not be available on a float column. So how close did we get? Let's look at the classical titanic data set to see some transformations.

```python
class TitanicPassenger(FeatureView):

    metadata = ... # Will present this later
    
    passenger_id = Entity(dtype=Int32())
    
    # Input values
    name = String()
    sex = String()
    survived = Bool()
    sibsp = Int32()
    cabin = String()
    
    # Fill nans with a constant `0` value
    age = Float().fill_na(0)

    has_siblings = sibsp != 0
    
    is_male, is_female = sex.one_hot_encode(['male', 'female'])
    ordinal_sex = sex.ordinal_categories(["male", "female"])

    # String operations that return a Bool.
    is_mr = name.contains('Mr.')
```


The above code removes all the need for source referencing and all strings, as shown by my `has_siblings = sibs != 0`. Furthermore, the type system knows which type is returned and helps the user with code completion and discovering new possibilities. We can also fill values if missing, like the age feature `age = Float().fill_na(0)`. The presented API is a good start, but what about the data management that other feature stores provide? So let's talk about documentation.

### Data documentation
Like other feature stores, we can also describe features and views, and it is all done by a simple `.description("...")`.

```python
class TitanicPassenger(FeatureView):

    metadata = FeatureViewMetadata(
        name="titanic",
        description="Features from the titanic dataset",
        ... # Will present this later
    ) 
    
    passenger_id = Entity(dtype=Int32())
    
    # Input values
    ...
    survived = Bool().description("If the passenger survived").
    sibsp = Int32().description("Number of siblings on titanic")
    
    # Fill nans with a constant `0` value
    age = Float().fill_na(0).description("A float as some have decimals")

    has_siblings = (sibsp != 0).description("Can not be negative, so if not 0 means they have a sibling")
```

Just see how elegant it is to have the description added. Kind of like a comment, but this can be used in other applications, like searching for features in something like a UI application. But notice, the `has_siblings` mention that the `sibsp` feature can not be under 0, but we use a 32 int data type. Such a constraint would be nice to validate.

### Data validation
Such constraints are the reason why we also have data validation easily accessible. And again, since a lower bound does not make sense on a string, the compiler will ensure this is not possible in the first place. So, where do we add such constraints?

```python
class TitanicPassenger(FeatureView):

    metadata = FeatureViewMetadata(
        name="titanic",
        description="Features from the titanic dataset",
        ... # Will present this later
    ) 
    
    passenger_id = Entity(dtype=Int32())
    
    # Input values
    ...
    survived = Bool().description("If the passenger survived").is_required()
    sex = String().accepted_values(["male", "female"])
    sibsp = Int32().description("Number of siblings on titanic").lower_bound(0)
    
    age = (
        Float()
            .fill_na(0)
            .description("A float as some have decimals")
            .lower_bound(0)
            .upper_bound(150)
    )

    has_siblings = (sibsp != 0).description("Can not be negative, so if not 0 means they have a sibling")
```
 
Again, having all this information close to each other makes the data behavior much clearer. Adding both lower and upper is required, and accepted values validation makes it much easier to test my data hypothesis faster. But where is the data located? Defining data locations is where data sources come into play.

### Data sources
We often create a model based on historical data sources, so let's define some data. But first, we can start with how to define a local CSV file and add that to the view.

```python
titanic_source = FileSource.csv_at("data/titanic.csv")

class TitanicPassenger(FeatureView):

    metadata = FeatureViewMetadata(
        name="titanic",
        description="Features from the titanic dataset",
        batch_source=titanic_source
    ) 
    
    passenger_id = Entity(dtype=Int32())
    
    ...
```
 
That's it. Just add a source to the `batch_source` in a view, and the library handles the rest. We can also connect to a PSQL source, Redshift, or an AWS file by changing the source.
 
```python
psql_source = PostgreSQLConfig("PSQL_URL").table("titanic")
redshit_url = RedshiftSQLConfig("REDSHIFT_URL").table("titanic")
aws_file = AwsS3Config(
    secret_token_env="SECRET", ...
).parquet_at("data-set/titanic.parquet")
```
 
The same thing applies to *stream sources*. Create a stream source from someplace, and add it to the `stream_source` field.

```python
redis_source = RedisConfig.localhost().stream_source(topic_name="titanic")
http_push = HttpStreamSource(topic_name="titanic")

class TitanicPassenger(FeatureView):

    metadata = FeatureViewMetadata(
        name="titanic",
        description="Features from the titanic dataset",
        batch_source=titanic_source,
        stream_source=redis_source
    ) 
```
 
An HTTP push with the `HttpStreamSource` may make more sense for those without streaming architecture. The HTTP push is not a very resilient method, as data can easily get lost, but it is an easy way of testing and requires little to no architecture.

### Models
Unlike Bender, this time, handling model training was not of priority. This priority was because I wanted better control of the foundational data. However, there was still interest in adding support for model use cases like a model. 

```python
titanic_model = Model(
    features=[
        TitanicPassenger.select(lambda view: [
            view.scaled_age,
            view.is_male,
            view.is_mr,
            view.has_siblings,
        ])
    ],
    targets=TitanicPassenger().survived
)
```

The shown case could be better, as only one feature view is used for a model, but you can combine multiple views. Such a view will define a model's input and know how to combine all the needed features efficiently. Such views can also simplify data set creation and enables a wide range of exciting features. However, you may need clarification on the `.select(lambda view: [...])` syntax? The reasoning is that this provides proper code completion and makes the whole system safer to use.

### Repository definition
The end goal of this system was to have a single source of truth that requires storing this information in some way. That is why all this information gets compiled into a schema stored in some defined location. This could be locally in a file, in a database, or even in an S3 bucket. Therefore making the information accessible by whoever needs it and fulfilling the single source of truth requirement as our batch, inference, and evaluation services can get the same features.

### Technology agnostic
As mentioned earlier, I tried avoiding using something like pickle. This has an exciting effect when loading the schema, as the programming language can be something other than Python and still work. For instance, as a fun project to learn Rust, I created a server that handles all feature pre-processing for a model given a JSON repo definition file as input, and the same thing can be done for any other technologies. This means that such a file can be used to unify batch-, stream- and edge-processing while still using different tech stacks. Furthermore, such a design enables the user to quickly change the processing engine out, based on their need. For instance, changing from pandas to Polars is one line of code.

```python
store = await FileSource.json_at(
    "feature-store.json"
).feature_store()
data = store.feature_view("titanic").all(limit=100)

pandas_df = await data.to_pandas()
polars_df = await data.to_polars()
```

This also means that it can adapt to the structure that you want. You can choose a separate Python- or Rust- server for pre-processing, the feature store integrated into your server's existing inference server, or a Kubernetes setup. The framework is not stopping you, and for the k8s structure, you may want even more functions.

### Use-case agnostic
Another intriguing effect of such a system is that it is use-case agnostic. This means the information can be used to process features for an ML model, as the following dataset shows.

```python
data = await store.model(
    "titanic_model"
).feature_for(
    entities
).test_split(
    0.1, target="survived"
).validation_split(
    0.1
).to_pandas()

# All the different datasets
print(data.train, data.test, data.validation)

# Get the X and y or also known as input and output
print(data.train.input, data.train.output)
```

But the same information can be used to set up data catalogs, monitoring systems, evaluation systems, and more. This is currently not implemented fully, but I have tested the theory by adding a model register to the model. Therefore making it possible to spin up pre-processing-, inference-, data catalogs- servers by providing a repository definition.

## The future
The library [Aligned](https://github.com/otovo/aligned) has proved its usefulness in Otovo. And I get inspired by tools like Ludwig, which tries to make end-to-end deep learning declarative, and KServe, which makes k8s management easier for ML use cases. Therefore, Aligned has just begun with what it can do. 
My mind is full of ideas for improving this library even more. However, thankfully we can stand on the shoulders of giants, as Aligned plans to not implement its own methods but instead rely on existing products. So what are some features to be planned in the near months?

### Aggregated features
The library currently does not support aggregated values. However, how to make this possible has been thought of and will be added soon. And as mentioned earlier, the interface for adding such values will be exquisite.

### Streaming setup
Aggregations values will need to align for batch and streaming processing. Therefore, this is why the framework will focus on simplifying both of them and adding better support for stream systems as a whole. Thus making streaming more natural and possible to use multiple technologies at once. So, for example, it makes sense to start with HTTP, then move to Redis Streams, and then move to Kafka as you scale. And this should not affect how you think of the product's logic.

### Inference servers
Even though there are more things on my mind, lastly, I would like to mention how setting up an inference server is part of the plans. All that is needed is some model register, and we can set up pre-processing and inference. Even with better streaming support, all inferences can be made seamlessly and automatically. Such a system will support all kinds of models, sklearn and TensorFlow, but also online learning models like River models.

## Conclusion
The Aligned library has innovated a new way to describe ML products. It defines a single source of truth for logic while keeping the technology stack flexible. Such innovation has been possible by removing the need to depend on a processing engine, leading to less- and more transparent- code. Furthermore, the declarative API has made it possible to comment, add data validation, and define feature transformation at the same location. Moreover, it leads to a precise definition of the intended result. Finally, the library allows it to fulfill multiple use cases such as stream- and batch-processing and set up data catalogs, monitoring systems, and complete inference servers. All because we define weak logical dependencies while leaving the details for later.

## Follow the development
So if you are interested in what aligned is trying to do, give it a star on [GitHub](https://github.com/otovo/aligned). And if you want to test it out briefly, all you need is to run the following code, and you have access to titanic features and the classical breast cancer dataset. But first run `pip install aligned`.

```python
from aladdin import FeatureStore, FileSource

os.environ['REDIS_URL'] = "redis://localhost:6379"

# The online store, which use the online source
online_store = await FileSource.from_path("https://raw.githubusercontent.com/otovo/aladdin-example/main/feature-store.json").feature_store()

# Write to the online store which stores the values in Redis
await online_store.feature_view("breast_scans_transformed")\
    .write({
        'scan_id': [1, 2],
        'area_mean': [1005, 1002],
        'compactness_mean': [0.23, 0.10],
        'perimeter_mean': [78, 90],
        'radius_mean': [20, 18],
        'smoothness_mean': [0.10, 0.2]
    })
    
# Read the values from Redis
processed_data = await online_store.features_for(
    { "scan_id": [1, 2] }, 
    features=["breast_scans_transformed:*"]
).to_df()
```