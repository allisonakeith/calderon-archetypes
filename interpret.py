#imports
#### Imports ####
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import AutoConfig, AutoModelForSequenceClassification


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoConfig, AutoModelForSequenceClassification

#import sns
import seaborn as sns

from transformers_interpret import MultiLabelClassificationExplainer

########################################## DEFINE INTERPRET FUNCTION ##########################################

def interpret_model(model, text, tokenizer, visualizer = False, id_num=0):
    cls_explainer = MultiLabelClassificationExplainer(
    model,
    tokenizer

    ## This is supposed to output custom lables
    # attribution_type: str = "lig",
    # custom_labels: List[str] | None = None
    )

    print("Interpreting input text ")

### Try to print all the visualizations into a single file 
    for sentence in text:
       word_attributions = cls_explainer(sentence[:511])
    if visualizer:
        cls_explainer.visualize("/results/viz_files/viz.html")
    return word_attributions
    

##################################################################################################


#### Define Model and Tokenizer ####
model_path = 'distilbert-base-multilingual-cased'  # Same path used during training
config = AutoConfig.from_pretrained(
    model_path,
    num_labels=6,
    hidden_dropout_prob=0.3,
    attention_probs_dropout_prob=0.3
)

model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the saved model weights
model.load_state_dict(torch.load('/pca_model.pth'))
model.eval()  # Set model to evaluation mode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


####################################################################


#### Load Data ####
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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


####################################################################

def tokenize_data(df, column_of_interest, tokenizer):
    encodings = tokenizer(list(df[column_of_interest]), truncation=True, padding=True, return_tensors='pt')

    # Create a PyTorch dataset
    class CustomDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

    dataset = CustomDataset(encodings, list(df['label']))
    return dataset

# Tokenize the full dataset
full_dataset = tokenize_data(df, 'tokens', tokenizer)

def extract_document_embeddings(model, dataset, device):
    model.eval()  # Set model to evaluation mode
    embeddings = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=16, shuffle=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            # Use the last hidden state CLS token embedding
            cls_embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
            embeddings.extend(cls_embeddings)
    return np.array(embeddings)

# Extract embeddings for your dataset
embeddings = extract_document_embeddings(model, full_dataset, device)



### INTERPRET MODEL ###
attributions = 1
        
 #create a list to store word attributions
word_attributions = []
text_list = []
#interpret the model
for index, row in comedias_df.iterrows():
            #### 
    text_list.append(row['tokens'])
    # print(text_list)

            ####
            #only take the first 512 tokens of the speech
    word_attribution = interpret_model(model, text_list, tokenizer, visualizer = False)
    print(word_attribution)
    word_attributions.append(word_attribution)

        #output word attributions to a file
word_attributions_df = pd.DataFrame(word_attributions)
word_attributions_df.to_csv(f"archetype-predict-pkg/results/january24_word_attributions.csv", index=False)

attributions+=1



#