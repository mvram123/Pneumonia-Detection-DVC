import json

# Initial Schema
with open("reports/metrics/scores.json", "w") as f:
    scores = {"model_scores": []}

    json.dump(scores, f, indent=4)

with open("reports/metrics/params.json", "w") as f:
    params = {"model_params": []}

    json.dump(params, f, indent=4)

with open("reports/metrics/metric.json", "w") as f:
    metric = {"model_metric": []}

    json.dump(metric, f, indent=4)
