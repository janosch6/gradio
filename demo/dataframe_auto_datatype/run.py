import gradio as gr
import pandas as pd
import numpy as np
import polars as pl
from datetime import datetime


rng = np.random.default_rng()

df_headers = ["String", "Int", "Float", "Pandas Time",
              "Numpy Time", "Datetime", "Boolean"]


list_data = [
    ["Irish Red Fox", 185000,   4.2, pd.Timestamp(
        '2017-01-01T12'), np.datetime64('now'), datetime(2022, 1, 1), True],
    ["Irish Badger", 95000,   8.5,  pd.Timestamp(
        '2018-01-01T12'), np.datetime64('now'), datetime(2023, 1, 1), True],
    ["Irish Otter", 13500,   5.5,  pd.Timestamp(
        '2025-01-01T12'), np.datetime64('now'), datetime(2024, 1, 1), False]
]
np_data = np.array(list_data, dtype=object)
pl_data = pl.DataFrame(list_data, schema=df_headers)

pd_data = pd.DataFrame(list_data, columns=df_headers)
styler_data = pd_data.style.apply(lambda row: [
                                  'background-color: lightgreen' if row['Boolean'] else '' for _ in row], axis=1)


print("Pandas: ", pd_data.dtypes)
print("Numpy: ", np_data.dtype)
print("Numpy: ", [str(np_data[:, i].dtype) for i in range(np_data.shape[1])])
print("Numpy: ", [str(type(np_data[0, i]).__name__)
      for i in range(np_data.shape[1])])
print("Polars: ", pl_data.dtypes)
print("Styler: ", styler_data.data.dtypes)
print("List: ", [type(val) for val in list_data[0]])


with gr.Blocks() as demo:

    df_type = gr.Radio(
        value="pandas",
        choices=["pandas", "numpy", "polars", "list", "styler"],
        label="Dataframe Type",
        type="value",
    )

    @gr.render(inputs=df_type)
    def render_df(df_type):
        if df_type == "pandas":
            df = gr.Dataframe(
                value=pd_data, headers=df_headers, interactive=True)
        elif df_type == "numpy":
            df = gr.Dataframe(
                value=np_data, headers=df_headers, interactive=True)
        elif df_type == "polars":
            df = gr.Dataframe(
                value=pl_data, headers=df_headers, interactive=True)
        elif df_type == "styler":
            df = gr.Dataframe(value=styler_data,
                              headers=df_headers, interactive=True)
        elif df_type == "list":
            df = gr.Dataframe(
                value=list_data, headers=df_headers, interactive=True)
        else:
            raise ValueError(f"Unsupported dataframe type: {df_type}")

        gr.Info(f"Dataframe Datatypes: {df.datatype}")
        return df


if __name__ == "__main__":
    demo.launch()
