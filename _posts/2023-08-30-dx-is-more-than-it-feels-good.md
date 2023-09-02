---
layout: post
title: "DX is more than 'It Feels Good'"
categories: [DX, UX, Software Design]
permalink: /posts/dx-is-more-than-it-feels-good
---

Many developers often lean towards tools that "feel good" and argue they have good Developer Experience (DX). However, this subjective measure of "feel good" is rooted in past experiences and familiarity with similar tools, making the argument of "good DX" worthless. Furthermore, relying solely on what feels comfortable can hinder innovation and limit our perspective. Instead, a more objective way of measuring DX and evaluating code quality will be needed if we continue using the term DX.

Therefore, using more tried-and-tested principles can benefit this debate. However, I am not proposing to use principles like [SOLID](https://en.wikipedia.org/wiki/SOLID), [DRY](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself), or [Clean Code](https://books.google.no/books/about/Clean_Code.html?id=hjEFCAAAQBAJ&source=kp_book_description&redir_esc=y); I'm instead proposing to use existing UX principles and apply them across different programming languages, frameworks, and paradigms.

But UX is for UI design, I hear you say.

And I would answer - no, it is not; it is much more flexible, so let me show you.

## Example 1: Pydantic vs dataclasses
Let's evaluate two codebases that fulfill the same need but use different implementations. Furthermore, we will determine why one solution is better using UX principles rather than the unclear "feel better."

The following code makes it possible to encode and load a JSON object. Our first implementation will use `dataclasses`, and the second will use `pydantic`.
Now, `dataclasses`'s use-case is not to make JSON easier to work with. However, `dataclasses` can implement our "happy path," so it could have been used for such a use-case.

So, let's look at the `dataclasses` implementation.

```python
from dataclasses import data class, asdict
import json

@dataclass
class Size:
    width: int
    height: int
    
size = Size(width=10, height=20)
size_json = json.dumps(asdict(size))
reconstructed = Size(**json.loads(size_json))
```

This is pretty straightforward. Declare a data class and its properties, create an instance of the object, convert it to a dict, and then to a JSON string. Lastly, decode the JSON string back to a dictionary and pass the fields to the constructor. 

So now is the `pydantic` solution.

```python
from pydantic import BaseModel
import json

class Size(BaseModel):
    width: int
    height: int

size = Size(width=10, height=20)
size_json = size.model_dump_json()
reconstructed = Size(**json.loads(size_json))
```

So, both solutions are very similar, and if I had gone with my "feeling," I would have implemented the `dataclasses` solution. 

Why? Because I do not like that we need to subclass from `BaseModel` in `pydantic`. Furthermore, `pydantic` is an external dependency, while `dataclasses` is already included in Python.

However, would this be the correct choice, and if so, why?

### Evaluate using UX
To evaluate our use case, I will mainly use [Nilsens Heuristics](https://www.nngroup.com/articles/ten-usability-heuristics/). These are very generic, but it is also what makes them so flexible. Furthermore, it is still better than "it feels good," so I think it is a step in the right direction.

However, using other UX principles like the [Proximity principle](https://www.nngroup.com/articles/gestalt-proximity/) - related information should be close to each other, which could be used to argue against C header files. Other concepts, e.g., from "[the design of everyday things](https://www.amazon.com/Design-Everyday-Things-Donald-Norman/dp/1452654123)" would also work.

### Error prevention
> Good error messages are important, but the best designs carefully prevent problems from occurring in the first place.
> 
> [NN Group](https://www.nngroup.com/articles/ten-usability-heuristics/)

While the `dataclasses` and `pydantic` solutions appear similar, their approaches to error handling are notably different. 

Our `dataclasses` solution allows the following invalid input.

```python
size = Size(width="10m", height="20cm")
```
Here, we send in a string while we expect an integer to be passed, and we can use this and send it into functions using the `Size` object.
However, `pydantic` will validate and catch such a case. Therefore, throwing an error if we do not match the expected data pattern prevents downstream problems.

Other examples that improve DX by preventing errors could be using something like Protocol Buffers rather than JSON for a similar reason.

### Visibility of system status
> The design should always keep users informed about what is going on through appropriate feedback within a reasonable amount of time.
> 
> [NN Group](https://www.nngroup.com/articles/ten-usability-heuristics/)

It can be harder to provide feedback in code as it is a static artifact. However, we can leverage tools like type systems, compilers, and linters, to fulfill this.

Furthermore, to continue our above example, let's add a new method.

```python
def area(size: Size) -> int:
    return size.width * size.height
```

Since `pydantic` throws when the data pattern is invalid, will `pydantic`, as a result, ensure our system state.
Therefore, only `pydantic` will make our system state more visible, as we know that our `Size` object contains the expected state that we want.

This is what I dislike about TypeScript, as we quickly get a false perception about our system state. Furthermore, showing the result of automated tests in a CI can also help present the system state.

### Help users recognize, diagnose, and recover from errors
> Error messages should be expressed in plain language (no error codes), precisely indicate the problem, and constructively suggest a solution.
> 
> [NN Group](https://www.nngroup.com/articles/ten-usability-heuristics/)

This is where the original UX heuristic may differ from "my" DX one. We sometimes need the technical jargon and details to fix the problem. However, the heuristic is still valuable, as precise error messages with guidance are super valuable.

Again, our `dataclasses` solution can fail if we call `area(size),` which can lead to the following error message:

```
TypeError: can't multiply sequence by non-int of type "str"
```
This tells us precisely what the problem is. However, we are told we can't do what we want, which is multiplying. Furthermore, the compounding effect of the mismatching system state makes it harder to understand why we can't multiply. The reason for this error is because of invalid input. 

However, our `pydantic` solution provides an error when trying to create an object. But this leads to the following error.

```
ValidationError: 2 validation errors for Size
width
  Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='10 m', input_type=str]
    For further information visit ...
height
  Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='20 cm', input_type=str]
    For further information visit ...
```

This message mentions where the error is located, what is wrong, how to fix it, and a link for more information.
It is more apparent that we have inputted invalid data, and most likely not due to poor code.

This guidance also makes Rust's compiler helpful, as it helps you locate and guide you in the correct solution when something is wrong.

With the presented UX principles in mind, my decision to go with what felt best would have been a poor decision, potentially leading to more frustration in the long run. 

However, this is one case; what about something that differs more?

## Example 2: Aligned vs Pandas
Let's look at another solution with data processing using an unconventional approach while presenting new heuristics. 

So, let's consider a scenario where we load geo data from a trip database, compute some features, and then use an ML model to predict the duration of the trip. We will use `pandas` and `aligned` to do the same thing. So, let's look at the `pandas` code.

```python
con = create_engine(os.environ["TAXI_DB_URL"])
df = pd.read_sql_query(
    sql="""
SELECT 
    id as trip_id, 
    
    pickup_latitude, 
    pickup_longitude,
    
    dropoff_latitude,  
    dropoff_longitude, 
    
    picked_up_at 
FROM trips;
""", 
    con=con
)

df["lat_diff"] = (df["pickup_latitude"] - df["dropoff_latitude"]) ** 2
df["long_diff"] = (df["pickup_longitude"] - df["dropoff_longitude"]) ** 2
df["travel_distance"] = (df["lat_diff"] + df["long_diff"]) ** 0.5

# The day in the month
df["day_of_week"] = df["picked_up_at"].dt.day

input = df[["day_of_week", "travel_distance"]]
duration = model.predict(input)
```

This is a common codebase for data applications. However, this solution has similar problems to our `dataclasses` solution earlier. 
We cannot validate our SQL code because it is written in a pure string. We do not make our code check against bad data, as we do not have data validation. This is intentionally not added, as presenting too much code in a blog post is uninteresting. Furthermore, experimental code can quickly skip data validation, as extra dependencies are needed, leading to odd errors.
Therefore, making our `df["picked_up_at"].dt.day` potentially error-prone, as we forcefully cast our value to a date time type without converting the `picked_up_at` to a proper value.

Furthermore, we can not use our Python [Language Server Protocol (LSP)](https://en.wikipedia.org/wiki/Language_Server_Protocol) to catch naming errors in our data frame, as we use strings to reference values. Leading to the potential error `KeyError: 'day_of_weak'` on runtime rather than an error on compile time. Similar to how we can not type check for values in dictionaries, but we can if described as a `class` object.

Therefore, we have little to no system status visibility, leading to less error prevention, and we do not help our users recognize, diagnose, and recover from errors.

A solution like `aligned` can significantly enhance the UX heuristics we've discussed. A quick introduction to `aligned.` Aligned is an ML tool used to make data more consistent at a system level. Therefore making it possible to standardize transformations in a technology-agnostic format. Making it possible to do ETL, even though it is not its primary purpose. 

So, how do `aligned` compare?

```python
taxi_db = PostgreSQLConfig(env_var="TAXI_DB_URL")

@feature_view(
    name= "trips,"
    description= "Features related to the departure of a taxi ride,"
    batch_source=taxi_db.table("trips", mapping_keys={"id": "trip_id"})
)
class Trips:
    trip_id = UUID().as_entity()

    picked_up_at = EventTimestamp()

    dropoff_latitude = Float().is_required()
    dropoff_longitude = Float().is_required()

    pickup_latitude = Float().is_required()
    pickup_longitude = Float().is_required()
    
    lat_diff = (pickup_latitude - dropoff_latitude) ** 2
    long_diff = (pickup_longitude - dropoff_longitude) ** 2
    travel_distance = (lat_diff + long_diff) ** 0.5
        
    day_of_week = picked_up_at.day.description("The day in the month")
    
    
input = await (Trips.query()
    .select(["travel_distance", "day_of_week"])
    .all()
    .to_pandas()
)
preds = model.predict(input)
```

Quickly, `aligned` makes it possible to declare data schemas and transformations using a class, enabling us to leverage the LSP and find errors earlier. Making it more straightforward which state our data is expected to be in and reducing errors by validating the anticipated state. Very similar to `pydantic`.

But let's compare using some new UX principles.

### Aesthetic and minimalist design
> Interfaces should not contain information that is irrelevant or rarely needed. Every extra unit of information in an interface competes with the relevant units of information and diminishes their relative visibility.
> 
> [NN Group](https://www.nngroup.com/articles/ten-usability-heuristics/)

The `pandas` solution contains fewer characters, which makes it more minimal in that sense, as we type less. However, the transformations in our `pandas` solution keep repeating `df["..."]`, which is irrelevant information. Therefore cluttering our logic and creating more noise. However, this is where `aligned` makes it way less noisy. Just look at how the two different lines differ below.

```python
# Pandas
df["lat_diff"] = (df["pickup_latitude"] - df["dropoff_latitude"]) ** 2

# Aligned
lat_diff = (pickup_latitude - dropoff_latitude) ** 2
```

Both do the same thing, but `aligned` makes it possible to remove all the `df` references while also removing the usage of `dict`-like data access. Therefore, our business logic is described with less noise `aligned`, and I would take less noise over fewer characters.

### Recognition rather than recall
> Minimize the user's memory load by making elements, actions, and options visible. The user should not have to remember information from one part of the interface to another.
> 
> [NN Group](https://www.nngroup.com/articles/ten-usability-heuristics/)

Furthermore, our `pandas` solution relies heavily on strings or a `dict` like access method. This has the unfortunate effect of hiding valuable information for our LSP. This results in less accurate code completions, leading to more cognitive load on the user, as a result, and as mentioned earlier. Our `pandas` code needs the user to remember that the `picked_up_at` is a date time and cast it to such a type `df["picked_up_at"].dt.day`.

However, `aligned` uses variables combined with defined types without casting. Therefore, helping our LSP provide valuable information about the types that each method returns and the potential paths our program can take. Enabling our users to search through our auto-completion and recognize possible programming paths. Therefore, recalling less than our `pandas` solution.

Again, this is why using the proximity principle can also help with recognizing more than recalling it.

### Consistency and standards
> Users should not have to wonder whether different words, situations, or actions mean the same thing.
> 
> [NN Group](https://www.nngroup.com/articles/ten-usability-heuristics/)

Following conventions and common standards is always good, making it easier to transition from different domains. One such standard could be the usage of SQL, using the standard operators as `+`, `-`, `*` etc., or using common names on data type as `Float,` `Int64`, and maybe not 'long long int.`

However, this is where `pandas` and `aligned` provide consistency and standards at different abstraction levels. As `pandas` provides data transformation consistency across different machine architectures, while `aligned` provides consistency at an application service level, transformations, and data dependencies can be shared in a serialized format. 

Therefore, they both provide consistency, depending on the intended use case. As a result, DX is not as easy as "this is better," but it is more a question about "it is better for this use-case." 

For, If we want to run one script that does data analytics occasionally, then `pandas` provides the consistency we want, as we can be confident the transformations will run on multiple types of machine hardware. However, if we need data transformations to be shared across employees or applications, then `pandas` may not provide good enough consistency, but `aligned` will.

Furthermore, this is where we can argue for DRY, as the whole point is to make our codebase more consistent when changes get implemented. However, will repeating the same code in two places be bad for consistency? Probably not, so maybe following the "Rule of three" rather than a pure DRY approach would be better.

### Flexibility and efficiency of use
> Shortcuts — hidden from novice users — may speed up the interaction for the expert user so that the design can cater to both inexperienced and experienced users.
> 
> [NN Group](https://www.nngroup.com/articles/ten-usability-heuristics/)

Furthermore, both solutions enable our users to speed up their work for common use cases. 
The `read_sql_query` makes connecting to an SQL db easier in `pandas`, rather than reading them manually and converting them into the desired format. However, `pandas` is intended to be used as a generic data processing framework and not for ML use cases.

Therefore, fulfilling needs specific to ML can be clunky to do with `pandas`, such as creating train, test, validation data sets, or setting upstream processing for low latency features.

As a result, `aligned` can streamline such use cases and make them easier, enabling one to opt into more flexible solutions when needed. That's why you see the `.to_pandas()` while also allowing you to set up a stream processing worker with the following code.

```python
from aligned.worker import StreamWorker
from aligned import RedisConfig, FileSource

definitions = FileSource.json_at("data-defintions/v1.json")

worker = StreamWorker.from_reference(
    definitions,
    sink_source=RedisConfig.localhost(),
).expose_metrics_at(port=8000)
```
Therefore, finding tools specializing in your needs will make you move faster. However, making it possible to opt into the underlying details will be helpful to make our tools flexible enough.

Furthermore, this is why I like the [fluent ORM](https://docs.vapor.codes/fluent/advanced/) created by the [Vapor team](https://docs.vapor.codes). As they provide:
A low-level raw SQL client.
A SQL query builder based on existing data classes.
A high-level ORM that fulfills common use cases.
It provides shortcuts while also enabling flexibility when needed.

### Latency
Lastly, even though this is not a heuristic, I would like to add performance as a criterion for DX, or how I would instead frame it, latency.

Furthermore, improved performance can affect cost. However, cost is more of a business side-effect, and I will not consider it affecting DX. I am only interested in how performance affects latency, as poor latency will degrade the developer experience, while good performance will be barely noticeable.

This happened to me at work, where we used `pandas` for most of our ETL. However, at some point, `pandas` lead to way too long run times. Therefore, switching to `polar` reduced the run time to 1 / 10, making the latency acceptable while avoiding complex `spark` clusters.

This is why `aligned` makes it possible to change the processing engine when needed. Therefore avoiding premature optimization, reducing costs, and making the solution performant.

In other words, high latency can ruin the developer experience, but only some will notice a more performant solution than an already good enough solution.

### Delving Deeper into Heuristics
While I've touched on several heuristics in this piece, there's a wealth more out there—like "Help and documentation", "User control and freedom", and "Match between system and the real world". I'd love to dive deeper, but we're already wading through quite a bit here. Perhaps in a future post!

## Conclusion
Developer Experience (DX) has been somewhat diluted to represent tools developers find comfortable. Therefore, it often means "what I am used to." But to drive valuable discussions and innovation, there's a pressing need to view DX through a more objective and comprehensive lens. Incorporating UX principles to evaluate code can pave the way for a more inclusive and critical approach, helping developers describe their code in new ways and make more educated choices for why technologies are good or bad. Even when they look foreign and odd at first glance.

If you found the read interesting, please share or give [Aligned](https://github.com/MatsMoll/aligned) a star. Thanks for reading.
