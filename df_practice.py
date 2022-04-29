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
series = series.sort_values(key = lambda col: col.apply(lambda x: str(x)))

series = pd.Series([100, 200, "python", 300.12, 400])
series = pd.concat([series, pd.Series([500, "php"])])

series = pd.Series(np.arange(10))
series_filtered = series[series <= 5]

x = pd.Series([1, 2, 3, 4, 5], index=["A", "B", "C", "D", "E"])
x = x.reindex(["B", "A", "C", "D", "E"])

x = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 3])
mu = x.mean()
std = x.std()

x = pd.Series([1, 2, 3, 4, 5])
y = pd.Series([2, 4, 6, 8, 10])
z = x[~x.isin(y)]
w = pd.concat([x[~x.isin(y)], y[~y.isin(x)]])

x = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
y = x.describe()

series = pd.Series([1, 7, 1, 6, 9, 1, 2, 9, 1, 2, 9, 2, 9, 0, 0])
counts = series.value_counts()

series[series != series.mode()[0]] = "Other"
import pdb; pdb.set_trace()
