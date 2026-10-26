import pandas as pd
import json
from pathlib import Path
from ucimlrepo import fetch_ucirepo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / 'data' / '01_raw'
DATA_PATH = RAW_DIR / 'adult.csv'
DICT_PATH = RAW_DIR / 'adult_dict.csv'
META_PATH = RAW_DIR / 'adult_metadata.json'

def ingest():
    if all([
        DATA_PATH.exists(),
        DICT_PATH.exists(),
        META_PATH.exists()
    ]):
        print('Data found in local directory. Ingestion skipped.')
        return

    print('Data not found in local directory. Fetching data...')

    # Create directory for raw data
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Fetch raw data
    adult = fetch_ucirepo(id=2)
    X = adult.data.features
    y = adult.data.targets

    # Merge data for export
    df = pd.merge(X, y, how='inner', left_index=True, right_index=True)

    # Export data to local directory
    df.to_csv(DATA_PATH, index=False)
    adult.variables.to_csv(DICT_PATH, index=False)
    with open(META_PATH, 'w') as metadata_file:
        json.dump(adult.metadata, metadata_file, indent=4)

    print(f"Data saved to {RAW_DIR}")



if __name__ == "__main__":
    ingest()