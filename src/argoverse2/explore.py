import numpy as np
import pandas as pd
import json
from pathlib import Path

if not hasattr(np, "bool"):
    np.bool = bool

from av2.datasets.motion_forecasting.scenario_serialization import load_argoverse_scenario_parquet
from av2.map.map_api import ArgoverseStaticMap

scenario_file = Path("/Users/aakarshrai/VSprojects/VFL/argoverse2_data/val/0a0ef009-9d44-4399-99e6-50004d345f34/scenario_0a0ef009-9d44-4399-99e6-50004d345f34.parquet")
map_file = Path("/Users/aakarshrai/VSprojects/VFL/argoverse2_data/val/0a0ef009-9d44-4399-99e6-50004d345f34/log_map_archive_0a0ef009-9d44-4399-99e6-50004d345f34.json")
scenario = load_argoverse_scenario_parquet(scenario_file)

# load map — use map_file here!
static_map = ArgoverseStaticMap.from_json(map_file)

print(type(scenario))
print(type(static_map))

print(scenario.keys)

