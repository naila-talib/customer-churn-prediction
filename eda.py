from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "customer_churn.csv"
GRAPH_DIR = BASE_DIR / "static" / "graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("CUSTOMER CHURN DATASET OVERVIEW")
print("=" * 60)
print("Shape:", df.shape)
print("\nColumns:")
for c in df.columns:
    print("-", c)

print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False).head(15))

if "Customer Status" in df.columns:
    print("\nCustomer Status:")
    print(df["Customer Status"].value_counts(dropna=False))

    counts = df["Customer Status"].value_counts()
    plt.figure(figsize=(7, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Customer Status Distribution")
    plt.xlabel("Customer Status")
    plt.ylabel("Customers")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "eda_customer_status.png", dpi=200)
    plt.close()

print("\nEDA completed.")
