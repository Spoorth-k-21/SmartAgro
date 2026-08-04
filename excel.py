import pandas as pd

p = r"data/New Final Data.xlsx"   # <-- correct path INSIDE your project
df = pd.read_excel(p)

print("Shape:", df.shape)
print("Columns:", df.columns)
print(df.head())
