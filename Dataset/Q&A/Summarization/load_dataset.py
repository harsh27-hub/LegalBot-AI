from datasets import load_dataset
import pandas as pd

ds = load_dataset("Exploration-Lab/IL-TUR", "summ")

train = pd.DataFrame(ds['train'])

test = pd.DataFrame(ds['test'])

train.to_csv("Dataset\\Summarization\\train.csv",index = False)
test.to_csv("Dataset\\Summarization\\test.csv",index=False)

# print(ds)

print("train columns: ",train.columns)
print("test columns: ",test.columns)