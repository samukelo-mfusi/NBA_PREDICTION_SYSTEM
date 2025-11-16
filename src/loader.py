import pandas as pd
from pathlib import Path
from utils import clean_basic

DATA = Path("data")
CLEAN = DATA / "cleaned"

def load_raw_file(filename):
    fpath = DATA / filename
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(fpath, sep=",", low_memory=False, encoding=encoding)
            print(f"Successfully loaded {filename} with {encoding} encoding")
            return clean_basic(df)
        except UnicodeDecodeError:
            continue
        except Exception:
            try:
                df = pd.read_csv(fpath, sep="|", low_memory=False, encoding=encoding)
                print(f"Successfully loaded {filename} with {encoding} encoding")
                return clean_basic(df)
            except:
                continue
    
    # Try with error handling if all encodings fail
    try:
        df = pd.read_csv(fpath, sep=",", low_memory=False, encoding='utf-8', errors='replace')
        print(f"Loaded {filename} with error replacement")
        return clean_basic(df)
    except:
        try:
            df = pd.read_csv(fpath, sep="|", low_memory=False, encoding='utf-8', errors='replace')
            print(f"Loaded {filename} with error replacement (pipe separator)")
            return clean_basic(df)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            return pd.DataFrame()

def save_clean(df, filename):
    CLEAN.mkdir(exist_ok=True)
    if not df.empty:
        df.to_csv(CLEAN / filename.replace(".txt", ".csv"), index=False)