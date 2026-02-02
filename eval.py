import torch
import torch.nn as nn
from model import Volatility_Prediction_Model
import matplotlib.pyplot as plt
from datasets import build_features,psi_split,time_series
import yfinance as yf

df1=yf.download("MSFT","2020-01-01")
df1 = yf.download("MSFT", start="2020-01-01")
df=build_features(df1)
train,test=psi_split(df)
x_test, y_test=time_series(test,30)

split=int(0.8*len(df))
test_df=df.iloc[split:,:]







device = "cuda" if torch.cuda.is_available() else "cpu"



# Rebuild model EXACTLY
model = Volatility_Prediction_Model(cnn_dim=49,n_embd=64, n_head=2, block_size=30, n_blocks=2).to(device)
state_dict = torch.load("model.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

print("Model loaded successfully.")

with torch.no_grad():
    y_pred=model(x_test.to(device)).squeeze(-1)
    y_pred=torch.exp(y_pred).cpu().numpy()

    

    #plotting
plt.figure(figsize=(12,4))

plt.plot(test_df.index, test_df['volatility_20'], label="True Volatility")

pred_index = test_df.index[30:30+len(y_pred)]
plt.plot(pred_index, y_pred, label="Predicted Volatility")

plt.legend()
plt.title("Real Volatility Prediction vs True Volatility")
plt.show()