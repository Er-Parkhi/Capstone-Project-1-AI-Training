
from pathlib import Path
 
import pandas as pd
 
SOURCE = Path("data/raw/onlinefraud.csv")

OUTPUT = Path("data/processed/fraud_sample.csv")

SAMPLE_FRACTION = 0.10

CHUNK_SIZE = 100_000

SEED = 42
 
 
def main():

    if not SOURCE.exists():

        raise FileNotFoundError(f"Dataset not found: {SOURCE}")
 
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
 
    total_rows = 0

    sampled_rows = 0

    first_chunk = True
 
    print(f"Source: {SOURCE}")

    print(f"Output: {OUTPUT}")

    print(f"Sampling {SAMPLE_FRACTION:.0%} of each chunk...")
 
    for chunk_number, chunk in enumerate(

        pd.read_csv(SOURCE, chunksize=CHUNK_SIZE, low_memory=False)

    ):

        total_rows += len(chunk)
 
        sample = chunk.sample(

            frac=SAMPLE_FRACTION,

            random_state=SEED + chunk_number,

        )
 
        sample.to_csv(

            OUTPUT,

            mode="w" if first_chunk else "a",

            header=first_chunk,

            index=False,

        )
 
        sampled_rows += len(sample)

        first_chunk = False
 
        print(

            f"Processed {total_rows:,} rows; "

            f"sampled {sampled_rows:,} rows",

            end="\r",

        )
 
    print("\n\nSampling complete.")

    print(f"Original rows: {total_rows:,}")

    print(f"Sample rows: {sampled_rows:,}")

    print(f"Output size: {OUTPUT.stat().st_size / (1024**2):.2f} MB")
 
    df = pd.read_csv(OUTPUT)

    print("\n--- SAMPLE TARGET DISTRIBUTION ---")

    print(df["isFraud"].value_counts().sort_index())

    print("\n--- SAMPLE FRAUD RATE ---")

    print(f"{df['isFraud'].mean() * 100:.5f}%")
 
    print("\n--- SAMPLE TRANSACTION TYPES ---")

    print(df["type"].value_counts())
 
 
if __name__ == "__main__":

    main()

