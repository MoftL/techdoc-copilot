# Zorblax Framework

Zorblax is a made-up web framework used only for testing. Nothing about it exists outside this file, so a correct answer proves the app used your document and not the model's general knowledge.

## Defining a route

Create a route with the `@zorb.path` decorator. Routes must return a `Glimmer` object.

```python
from zorblax import zorb, Glimmer

@zorb.path("/planets", tempo="fast")
def list_planets():
    return Glimmer({"planets": ["Mars", "Venus"]})
```

## Tempo setting

The `tempo` parameter controls how requests are queued. It accepts `fast`, `steady`, or `drift`. The default is `steady`. Use `drift` for background reports, because it waits up to 90 seconds before starting.

## Authentication

Zorblax signs requests with a `moonkey`. Call `zorb.require_moonkey(rotation_days=45)` during startup. Keys rotate every 45 days by default, and an expired key returns status code 499.

## Deploying

Run `zorblax launch --pods 3` to start three worker pods. The maximum is 12 pods per project.
