# -*- coding: utf-8 -*-
import pandas as pd

def get_cat_col(df: pd.DataFrame, df_label:str = None, fillna:bool = False) -> []:
    categorical_columns = []
    for col, col_type in df.dtypes.iteritems():
        if col_type == 'O':
            categorical_columns.append(col)
        elif fillna and col_type != 'O':
            df[col].fillna(0, inplace=True)
            
    out_msg = "\nCategorical columns found"
    if df_label:
       out_msg += "for {df_label}"
    print(out_msg)
    print("=" * len(out_msg))
    
    for col in categorical_columns:
        print(f"\t- {col}")
    return categorical_columns

def disp_col_w_missing(df: pd.DataFrame, df_label:str = None, colList:[] = None, threshold:int = 0):
    col_w_missing = None
    if colList:
        col_w_missing = pd.DataFrame(df[colList].isnull().sum())
    else:
        col_w_missing = pd.DataFrame(df.isnull().sum())
    
    out_msg = "\nSum of Missing Values"
    if df_label:
        out_msg += f" in {df_label}"
    
    print(out_msg)
    print("=" * len(out_msg))
    print(col_w_missing[col_w_missing.iloc[:,0] > 0])
    
    
def gen_json_testset(df: pd.DataFrame, rando_num:int = 0, no_status:bool = False):
    local_df = df
    if no_status:
        local_df = df.drop("Status", axis=1)
    if rando_num:
        rando_df = local_df.sample(n=rando_num)
        return rando_df.to_json(orient="records")
    return local_df.to_json(orient="records")