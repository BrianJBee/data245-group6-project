import pandas as pd
import json
import pandas as pd
from pathlib import Path
from folktables import ACSDataSource, ACSIncome

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / 'data' / '01_raw'
DATA_PATH = RAW_DIR / 'folktables_ca.csv'
MASTER_PATH = RAW_DIR / 'master_data'

def ingest() -> pd.DataFrame:
    if DATA_PATH.exists():
        print('Data found in local directory. Ingestion skipped.')
        df = pd.read_csv(DATA_PATH)
        return (df)

    print('Data not found in local directory. Fetching data...')

    # Create directory for raw data
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Fetch raw data
    data_source = ACSDataSource(
        survey_year='2018',
        horizon='1-Year',
        survey='person',
        root_dir=str(MASTER_PATH)
    )
    data = data_source.get_data(states=['CA'], download=True)

    X, y, group = ACSIncome.df_to_pandas(data)

    # Merge data for export
    df = pd.merge(X, y, how='inner', left_index=True, right_index=True)

    # Export data to local directory
    df.to_csv(DATA_PATH, index=False)
    print(f"Data saved to {DATA_PATH}")

    # Re-import for return consistency
    df = pd.read_csv(DATA_PATH)
    return df



if __name__ == "__main__":
    ingest()