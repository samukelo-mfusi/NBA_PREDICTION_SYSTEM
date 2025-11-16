import pandas as pd
import numpy as np

def remove_unnamed(df):
    return df.loc[:, ~df.columns.str.contains("Unnamed")]

def to_numeric(df):
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="ignore")
    return df

def clean_basic(df):
    df = remove_unnamed(df)
    df = df.dropna(how="all")
    df = df.dropna(how="all", axis=1)
    df = to_numeric(df)
    return df
