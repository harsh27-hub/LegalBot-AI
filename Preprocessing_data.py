# Script to remove outliers with tokens more than the limit of the model

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_dataset = pd.DataFrame(pd.read_csv(r"Dataset\Summarization\train.csv"))
test_dataset = pd.read_csv(r"Dataset\Summarization\test.csv")

data1 = pd.DataFrame([train_dataset['id'],train_dataset["document"],train_dataset["summary"],train_dataset["num_doc_tokens"], train_dataset["num_summ_tokens"]]).T

data1.columns = ["ids","document","summary","doc_size","summ_size"]

data2 = pd.DataFrame([test_dataset['id'],test_dataset["document"],test_dataset["summary"],test_dataset["num_doc_tokens"], test_dataset["num_summ_tokens"]]).T

data2.columns = ["ids","document","summary","doc_size","summ_size"]

# sns.boxplot(data['doc_size'])
# plt.show()

#sns.boxplot(data2['doc_size'])
#plt.show()

# Filter out the outliers
filtered_data1 = data1[(data1['doc_size'] >= 0) & (data1['doc_size'] <= 7000)]

filtered_data2 = data2[(data2['doc_size'] >= 0) & (data2['doc_size'] <= 7000)]

# Plot the boxplot for the filtered data
sns.boxplot(filtered_data1['doc_size'])
# plt.show()

# print(data2.describe())
filtered_data1 = filtered_data1.drop(columns=["doc_size","summ_size"])
filtered_data2 = filtered_data2.drop(columns=["doc_size","summ_size"])
print(filtered_data2.head()) 
print(filtered_data1.head())

filtered_data1.to_csv(r"Usable_Dataset\Summarization\train.csv")

filtered_data2.to_csv(r"Usable_Dataset\Summarization\test.csv")



