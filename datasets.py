import pandas as pd
import numpy as np
import yfinance as yf
import torch
from sklearn.linear_model import LinearRegression

# Flatten columns
def build_features(df:pd.DataFrame):
    
        df.columns = df.columns.get_level_values(0)
        df=df.astype(float)

        #  RETURNS


        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        df_ex=df["log_return"].rolling(20).std()
        df["simple_return"] = df["Close"].pct_change()


        #PRICE STRUCTURE
        # =========================
        df["range"] = df["High"] - df["Low"]
        df["body"] = df["Close"] - df["Open"]
        df["upper_wick"] = df["High"] - np.maximum(df["Open"], df["Close"])
        df["lower_wick"] = np.minimum(df["Open"], df["Close"]) - df["Low"]

        df["gap_return"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)


        # TREND / MOMENTUM

        df["momentum_5"] = df["Close"] / df["Close"].shift(5) - 1
        df["momentum_10"] = df["Close"] / df["Close"].shift(10) - 1

        df["ema_12"] = df["Close"].ewm(span=12).mean()
        df["ema_26"] = df["Close"].ewm(span=26).mean()
        df["macd"] = df["ema_12"] - df["ema_26"]


        # MEAN REVERSION

        df["sma_20"] = df["Close"].rolling(20).mean()
        df["ma_dev_20"] = df["Close"] - df["sma_20"]
        df["zscore_20"] = df["ma_dev_20"] / df["Close"].rolling(20).std()
        df["vol_lag_1"] = df_ex.shift(1)
        df["vol_lag_5"] = df_ex.shift(5)
        df["vol_lag_20"] = df_ex.shift(20)


        # VOLATILITY (TARGET)




        # VOLUME / LIQUIDITY

        df["volume_change"] = df["Volume"].pct_change()
        df["volume_norm"] = df["Volume"] / df["Volume"].rolling(20).mean()
        df

        #MARKET STRESS

        df["range_norm"] = df["range"] / df["range"].rolling(20).mean()
        df["volatility_20"] = df["log_return"].rolling(20).std()

        # CLEAN DATA

        df = df.dropna()
        df.iloc[:,:5]=(df.iloc[:,:5]-df.iloc[:,:5].mean(axis=0))/df.iloc[:,:5].std(axis=0)
        return df



def psi_split(df):
    split=int(0.8*len(df))
    lr=LinearRegression()
    dft=df.iloc[:split,:].copy()
    dft2=df.iloc[:split,:].copy()
    dft3=df.iloc[split:,:].copy()
    dft["v_t+1"]=dft["volatility_20"].shift(-1)
    dft=dft.dropna()
    lr.fit(dft.iloc[:,5:-1].to_numpy(),dft.iloc[:,-1].to_numpy())
    X=dft2.iloc[:,5:].to_numpy()
    X1=dft3.iloc[:,5:].to_numpy()
    x1,x2=X*lr.coef_+lr.intercept_,X1*lr.coef_+lr.intercept_
    train=np.concatenate([dft2.to_numpy(),x1],axis=1)
    test=np.concatenate([dft3.to_numpy(),x2],axis=1)
    return train,test

def time_series(data,seq_len):
            df_m=data.copy()
            target=data[:,-1].copy()
            seq_len=30
            data=np.array([df_m[i:i+seq_len] for i in range(len(df_m)-seq_len)])
            lab=np.array([target[i+seq_len] for i in range(len(target)-seq_len)])
            return torch.from_numpy(data).float(),torch.from_numpy(lab).float()
