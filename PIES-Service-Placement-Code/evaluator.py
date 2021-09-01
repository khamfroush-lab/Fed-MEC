# -- coding: future_fstrings --
import os
import pandas as pd
import warnings

from collections import defaultdict
from src.services.torch_eval import *

warnings.filterwarnings("ignore")

METRICS_DIR = os.path.join("out", "service_metrics")

if __name__ == "__main__":
    data = imagenet_data(do_preprocessing=True)

    metrics = defaultdict(list)
    s = Services.IMAGE_CLASSIFICATION
    n = None

    for m in Services.get_models(s):
        model_name = Services.get_model_name(s, m)
        model = Services.get_service_model_from_str(model_name)
        acc, cpu, delay = imagenet_evaluate(model, first_n=n, pbar_desc=model_name)
        metrics["service"].append("img-class")
        metrics["model"].append(model_name)
        metrics["s"].append(s)
        metrics["m"].append(m)
        metrics["acc"].append(acc)
        metrics["cpu"].append(cpu)
        metrics["comp_delay"].append(delay)

    metrics_filename = os.path.join(METRICS_DIR, "new_metrics.csv")
    metrics_df = pd.DataFrame.from_dict(metrics)
    metrics_df.to_csv(metrics_filename)
