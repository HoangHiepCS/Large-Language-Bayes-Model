import json

with open("experiment_results_anant/all_results.json") as f:
    all_results = json.load(f)

models = all_results[0]['full_result']['model_codes']
print(models[0])
print("xxxxxx")
print(models[65])
print("xxxxxx")
print(models[140])