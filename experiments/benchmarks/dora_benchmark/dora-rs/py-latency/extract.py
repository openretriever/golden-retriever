import numpy as np
import pandas as pd
import pyarrow as pa

LOOKBACK = "15m"
SERVICE = "op_2"
LIMIT = "400"
JAEGER_API = f"http://localhost:16686/api/traces?service={SERVICE}&lookback={LOOKBACK}&limit={LIMIT}"

df = pd.read_json(JAEGER_API)
df = pd.DataFrame(df.data.values.tolist())
df_spans = pd.DataFrame(df.spans.values.tolist())

with pa.memory_map("latency.arrow", "r") as source:
    df_data = pa.ipc.open_file(source).read_all().to_pandas()

df = df.merge(df_data, left_on="traceID", right_on="trace_id")
df["len"] = df["latency"].map(len)
df["log_len"] = df["len"].map(np.log10)


def get_diff_time(row):
    tmp_df = pd.DataFrame(row)
    tmp_df["end_time"] = tmp_df["startTime"] + tmp_df["duration"]
    # Compute difference between one row end_time and next row start_time
    tmp_df["diff_time"] = tmp_df["startTime"].shift(-1) - tmp_df["end_time"]
    return tmp_df["diff_time"][0]


def get_send_output_time(row):
    tmp_df = pd.DataFrame(row)
    output = tmp_df[tmp_df["operationName"] == "send_output"]["duration"]
    return output.values


df["lat"] = df["spans"].map(get_diff_time)
df = df[df["send_output_latency"].map(len) > 0]

df["send_output_latency"] = df["spans"].map(get_send_output_time)

df["lat_1"] = df["lat"].map(lambda x: x[1])
df["lat_2"] = df["lat"].map(lambda x: x[2])
df.plot.scatter(x="log_len", y="send_output_latency")

import matplotlib.pyplot as plt

plt.show()
