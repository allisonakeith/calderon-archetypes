import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import ast


from split import split_data
from test_train import test_model, train_model


from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import AutoConfig, AutoModelForSequenceClassification



comedias_df = pd.read_csv('/archetype-predict-pkg/all_characters.csv')

#identify data we want to work with
character_types = ['dama','criado', 'galán' , 'criada' , 'rey' , 'reina']
comedias_df = comedias_df[comedias_df['archetype'].isin(character_types)]

comedias_df = comedias_df[comedias_df['words_spoken'] > 50]

# comedias_df['tokens'] = comedias_df['tokens'].str.slice(0,512)

print(comedias_df.head)
print(comedias_df['archetype'].value_counts())

#from https://towardsdatascience.com/multi-class-text-classification-with-deep-learning-using-bert-b59ca2f5c613
# transform the archetypes into numeric variables

possible_labels = comedias_df.archetype.unique()

label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index
print(label_dict)

df = comedias_df
df['label'] = df.archetype.replace(label_dict)


##################################################
model_path =  'distilbert-base-multilingual-cased' #'dccuchile/bert-base-spanish-wwm-cased' 
model_name = 'distilbert-multilingual' #'bert-base-spanish' #


# dropout #
config = AutoConfig.from_pretrained(
    model_path,
    num_labels=6,
    hidden_dropout_prob=0.3,  # Default is 0.1; increase for more regularization
    attention_probs_dropout_prob=0.3,
    random_seed = 42
)

model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)


num_epochs = 40
lr = 0.00001
batch_size = 12



# # Define optimizer, loss is already specified as crossEntropyLoss in the model


class_weights = torch.tensor([1.0, 1.0, 1.2, 1.5, 1.8, 2.0]).to(device) #this doesn't get used in the model


######### SET DROPOUT TO REDUCE OVERFITTING #############


optimizer = Adam(model.parameters(), lr=lr)
#optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)  #



##add a learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)



model.to(device)


    


train_dataset, test_dataset, train_data, test_data, val_dataset, val_data = split_data(df, 'tokens', tokenizer)
trained_model, best_epoch = train_model(model, train_dataset, val_dataset, num_epochs, batch_size, optimizer, device, pos_weight=class_weights)

accuracy, all_predictions, all_probabilities = test_model(trained_model, test_dataset, batch_size, device)
                            
test_data['predictions'] = all_predictions
test_data['probabilities'] = all_probabilities

print(all_predictions)
print("WRITING FILE")
file_name = f"archetype-predict-pkg/results/jan23.csv"
test_data.to_csv(file_name, index=False)


# ### INTERPRET MODEL ###


# attribution = 1
        
#  #create a list to store word attributions
# word_attributions = []
# text_list = []
# #interpret the model
#         for index, row in val_data.iterrows():
#             #### 
#             text_list.append(row['tokens'])

#             ####
#             #only take the first 512 tokens of the speech
#             word_attribution = interpret_model(trained_model, text_list, tokenizer, visualizer = False)
#             word_attributions.append(word_attribution)

#         #output word attributions to a file
#         word_attributions_df = pd.DataFrame(word_attributions)
#         word_attributions_df.to_csv(f"wp1-semantic-analysis/gender-predict-pkg/results/june7_{column_of_interest}_word_attributions.csv", index=False)
#         attribution += 1
