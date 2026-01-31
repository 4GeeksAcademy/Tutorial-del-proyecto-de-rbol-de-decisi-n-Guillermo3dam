from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np




# load the .env file variables
load_dotenv()


def db_connect():
    import os
    engine = create_engine(os.getenv('DATABASE_URL'))
    engine.connect()
    return engine


def missing_values_summary(df):

    missing_count = df.isnull().sum().loc[lambda x: x > 0]

    if missing_count.empty:
        print("No missing values were found in the dataset.")
        return pd.DataFrame()
    
    missing_percent = round((missing_count / len(df)) * 100, 2)

    print("List of missing values:")

    missing_df = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing Percent (%)": missing_percent
    }).sort_values(by="Missing Count", ascending=False)

    return missing_df

def unknown_values_summary(df, unknown_value):

    unknown_count = (df == unknown_value).sum().loc[lambda x: x > 0]

    if unknown_count.empty:
        print(f"There are no {unknown_value} values in the dataset.")
        return pd.DataFrame()

    unknown_percent = round((unknown_count / len(df)) * 100, 2)

    print(f"List of {unknown_value} values:")

    unknown_df = pd.DataFrame({
        unknown_value + " Count": unknown_count,
        unknown_value + " Percent (%)": unknown_percent
    }).sort_values(by=unknown_value + " Count", ascending=False)

    return unknown_df


def remove_duplicates(df, id = None):

    filt_df = df.copy()

    if id and id in filt_df.columns:
        subset_cols = filt_df.columns.difference([id])
    else:
        subset_cols = None

    num_duplicates = filt_df.duplicated(subset=subset_cols).sum()

    if id and id in filt_df.columns:
        print(f"Number of duplicate rows (excluding '{id}'): {num_duplicates}")
    else:
        print(f"Number of duplicate rows: {num_duplicates}")

    if num_duplicates > 0:
        filt_df = filt_df.drop_duplicates(subset=subset_cols)
        print(f"Duplicates removed. The new shape is: {filt_df.shape[0]} rows and {filt_df.shape[1]} columns")
    else:
        print("No duplicates found.")

    return filt_df

def factorize_categorical(df):

    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    if not categorical_cols:
        print("No categorical columns found. Returning original DataFrame copy.")
        return df.copy()
    
    print("Categorical cols:", categorical_cols)

    total_data = df.copy()

    for col in categorical_cols:
        total_data[f"{col}_n"], uniques = pd.factorize(total_data[col])

        rules_dict = {
            value: int(code) for code, value in enumerate(uniques)
        }

        with open(f"../data/interim/rules_transformation_{col}.json", "w") as f:
            json.dump(rules_dict, f, indent=4)

    return total_data


def plot_correlation_heatmap(df):
    corr_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.show()


def replace_outliers(column, df):
  stats = df[column].describe()
  iqr = stats["75%"] - stats["25%"]
  upper_limit = stats["75%"] + 1.5 * iqr
  lower_limit = stats["25%"] - 1.5 * iqr

  if lower_limit < 0: lower_limit = min(df[column])
  
  outliers_exist = ((df[column] < lower_limit) | (df[column] > upper_limit)).any()

  if outliers_exist:
    original = df[column].copy()
    df[column] = df[column].clip(lower_limit, upper_limit)
    n_modified = (original != df[column]).sum()
    print(f"\n{n_modified} values clipped in {column}")
    print(f"Lower limit: {lower_limit}, Upper limit: {upper_limit}, IQR: {iqr}")

    
  else:
    print(f"\nNo outliers detected in {column}. No changes made.")


  return df, {
    "lower_limit": round(float(lower_limit), 3),
    "upper_limit": round(float(upper_limit), 3)
  }


def fill_missing_values(df, median_cols=None, mode_cols=None, mean_cols=None):
    df_copy = df.copy()

    # Cuando una variable es entera pero numérica/incontable
    if median_cols:
        for col in median_cols:
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
   
    # Cuando una variable es categórica
    if mode_cols:
        for col in mode_cols:
            df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])

    # Cuando una variable es puramente decimal
    if mean_cols:
        for col in mean_cols:
            df_copy[col] = df_copy[col].fillna(df_copy[col].mean())

    return df_copy


def replace_unknown_with_nan(df, cat_cols, rules_path="../data/interim/"):
    df_copy = df.copy()

    for col in cat_cols:
        col_n = f"{col}_n"
        file_path = f"{rules_path}/rules_transformation_{col}.json"
        with open(file_path, "r") as f:
            rules = json.load(f)

        if "unknown" in rules:
            unknown_code = rules["unknown"]
            df_copy[col_n] = df_copy[col_n].replace(unknown_code, np.nan)

    return df_copy


def check_missing_values(df):
    missing_count = df.isna().sum().loc[lambda x: x > 0]

    if missing_count.empty:
        print("No missing values were found in the dataset.")
        return pd.DataFrame()
    
    missing_percent = round((missing_count / len(df)) * 100, 2)
    
    print("List of missing values:")

    missing_df = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing Percent (%)": missing_percent
    }).sort_values(by="Missing Count", ascending=False)

    return missing_df

def merge_columns(df_with_outliers, df_without_outliers, merge, first_element, second_element):

    df_with_outliers[merge] = df_with_outliers[[first_element, second_element]].max(axis=1)
    df_without_outliers[merge] = df_without_outliers[[first_element, second_element]].max(axis=1)

    return df_with_outliers, df_without_outliers

    """df_with_outliers[merge] = df_with_outliers[first_element] + df_with_outliers[second_element]
    df_without_outliers[merge] = df_without_outliers[first_element] + df_without_outliers[second_element]
    
    return df_with_outliers, df_without_outliers"""