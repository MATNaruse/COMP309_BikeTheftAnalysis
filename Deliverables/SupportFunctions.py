# -*- coding: utf-8 -*-
"""
COMP309 Bike Theft Analysis
===========================
301 063 251 : Arthur Batista
300 549 638 : Matthew Naruse
301 041 132 : Trent B Minia
300 982 276 : Simon Ducuara
300 944 562 : Zeedan Ahmed
"""
import pandas as pd
import datetime

def get_cat_col(df: pd.DataFrame, df_label:str = None, fillna:bool = False) -> []:
    """
    

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.
    df_label : str, optional
        DESCRIPTION. The default is None.
    fillna : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    []
        DESCRIPTION.

    """
    categorical_columns = []
    for col, col_type in df.dtypes.iteritems():
        if col_type == 'O':
            categorical_columns.append(col)
        elif fillna and col_type != 'O':
            df[col].fillna(0, inplace=True)
            
    out_msg = "\nCategorical columns found"
    if df_label:
       out_msg += " for " + df_label
    print(out_msg)
    print("=" * len(out_msg))
    
    for col in categorical_columns:
        print(f"\t- {col}")
    return categorical_columns


def disp_col_w_missing(df: pd.DataFrame, df_label:str = None, colList:[] = None, threshold:int = 0):
    """
    

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.
    df_label : str, optional
        DESCRIPTION. The default is None.
    colList : [], optional
        DESCRIPTION. The default is None.
    threshold : int, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    """
    col_w_missing = None
    if colList:
        col_w_missing = pd.DataFrame(df[colList].isnull().sum())
    else:
        col_w_missing = pd.DataFrame(df.isnull().sum())
    
    out_msg = "\nSum of Missing Values"
    if df_label:
        out_msg += " for " + df_label
    
    print(out_msg)
    print("=" * len(out_msg))
    print(col_w_missing[col_w_missing.iloc[:,0] > 0])
    

def parse_time(timeList: []) -> []:
    """
    

    Parameters
    ----------
    timeList : []
        DESCRIPTION.

    Returns
    -------
    []
        DESCRIPTION.

    """
    # For parsing 'Occurrence_Time' into calc. minutes
    out_list: [] = []
    dt_delta = datetime.datetime(1900,1, 1)
    for time in timeList:
        temp_time_hold = datetime.datetime.strptime(time, '%H:%M')
        out_list.append((temp_time_hold - dt_delta).total_seconds()/60)
    return out_list
    

def gen_json_testset(df: pd.DataFrame, rando_num:int = 0, no_status:bool = False) -> str:
    """
    

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.
    rando_num : int, optional
        DESCRIPTION. The default is 0.
    no_status : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    str
        DESCRIPTION.

    """
    local_df = df
    if no_status:
        local_df = df.drop("Status", axis=1)
    if rando_num:
        rando_df = local_df.sample(n=rando_num)
        return rando_df.to_json(orient="records")
    return local_df.to_json(orient="records")


def gen_json_dummy(df: pd.DataFrame, pinColValue: [], size:int, toJson:bool = False) -> any:
    """
    

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.
    pinColValue : []
        DESCRIPTION.
    size : int
        DESCRIPTION.
    toJson : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    any
        DESCRIPTION.

    """
    local_df = df.sample(size)
    colValDict: {} = {}
    for tup in pinColValue:
        colValDict[tup[0]] = tup[1]
    
    for key, val in colValDict.items():
        print(f"{key} | {val}")
        local_df[key].replace(regex="^.*$", value=val, inplace=True)
    
    if toJson:
        local_df = local_df.drop('Status', axis=1)
        return local_df.to_json(orient="records")
    return local_df

    
    
    