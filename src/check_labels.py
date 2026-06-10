import pandas as pd

df = pd.read_csv(
    "data/shopsense_examples.csv"
)

print(
    df["esci_label"].value_counts()
)