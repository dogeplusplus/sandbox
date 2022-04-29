import pandas as pd
import numpy as np

x = np.arange(1, 10, 1)
df = pd.DataFrame({"x": x})

values = df.x.to_list()

x = np.random.randint(0, 10, (100))
y = np.random.randint(0, 10, (100))
df = pd.DataFrame({"x": x, "y": y})
z = df.y - df.x
w = df.y < df.x

items = {"a": 100, "b": 200, "c": 300, "d": 400, "e": 800}
df = pd.Series(items)

series = pd.Series([100, 200, "python", 300.12, 400])
series_converted = pd.to_numeric(series, errors="coerce")


df = pd.DataFrame({
    "col1": [1, 2, 3, 4, 7, 11],
    "col2": [4, 5, 6, 9, 5, 0],
    "col3": [7, 5, 8, 12, 1, 11],
})
series = df["col1"]
numpy = series.to_numpy()


series_unflattened = pd.Series([["Red", "Green", "White"], ["Red", "Black"], ["Yellow"]])
series = series_unflattened.apply(pd.Series).stack().reset_index(drop=True)

series = pd.Series([100, 200, "python", 300.12, 400])
sorted_series = series.astype(object).sort_values()
series.sort_values(key = lambda col: col.apply(lambda x: str(x)))
