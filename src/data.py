# src/data.py
import pandas as pd
from sklearn.datasets import load_iris
from pathlib import Path

out = Path("data")
out.mkdir(exist_ok=True)
iris = load_iris(as_frame=True)
df = pd.concat([iris.data, iris.target.rename("target")], axis=1)
df.to_csv(out/"iris.csv", index=False)
print("wrote data/iris.csv")
