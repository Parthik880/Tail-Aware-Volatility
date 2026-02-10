
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import yfinance as yf
import math
from datasets import build_features,time_series,psi_split

#ticker and #date
df=build_features()
train,test=psi_split(df)

split=int(0.8*len(df))
#train





x_train,y_train=time_series(train,30)

x_test, y_test=time_series(test,30)


from model import Volatility_Prediction_Model

device='cuda' if torch.cuda.is_available() else 'cpu'
model = Volatility_Prediction_Model(n_embd=64, n_head=2, block_size=30, n_blocks=3).to(device)
#model=torch.compile(model)


def main(k):
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)


        # DataLoaders

        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        




        #training loop




        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        loss_fn = nn.MSELoss()


        from tqdm import trange
        for epoch in trange(30,desc="training",unit="steps"):
            model.train()
            train_loss=0
            for xb,yb in train_loader:
                optimizer.zero_grad()
                xb = xb.to(device)
                yb = yb.to(device)
                yb = torch.log(yb + 1e-8)
                pred = model(xb).squeeze(-1)
                mse = loss_fn(pred, yb)
                tail_penalty = ((yb - pred).relu())**2
                pred_real = torch.exp(pred)
                rel_loss = ((pred_real - yb) / (yb + 1e-6))**2


                loss = mse + k * tail_penalty.mean()+ 0.1 * rel_loss.mean()
                loss /= len(train_loader)
                train_loss += loss.item()

                

                

                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
            
                optimizer.step()
                #rain_loss += loss.item()
            #train_loss /= len(train_loader)
            model.eval()
            test_loss=0
            real_vol_loss=0
            with torch.no_grad():
                    for xb, yb in test_loader:
                        xb = xb.to(device)
                        yb = yb.to(device)
                
                        # log target
                        yb_log = torch.log(torch.clamp(yb, min=1e-6))
                
                        # predict log-vol
                        pred = model(xb).squeeze(-1)
                
                        # log-vol loss
                        mse = loss_fn(pred, yb_log)
                        tail_penalty = ((yb_log - pred).relu())**2
                        loss = mse + k * tail_penalty.mean()
                        loss /= len(test_loader)
                        test_loss += loss.item()
                
                        # real-vol loss
                        pred_real = torch.exp(pred)
                        real_vol_loss += loss_fn(pred_real, yb).item()
                


            
            #test_loss /= len(test_loader)
            real_vol_loss /= len(test_loader)

            
            if epoch%5==0:

                print(f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} | real_vol_loss: {real_vol_loss:.5f}")

        torch.save(model.state_dict(), "model.pth")
        print("Model saved")

k=1.8 #hyperparameter between 1 to 3
if __name__ == "__main__":
        main(k)
print(f"Volatility Prediction Model Trained with k={k}")
