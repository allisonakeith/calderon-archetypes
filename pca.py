#### Imports ####
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from test_train import test_model

from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import AutoConfig, AutoModelForSequenceClassification


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoConfig, AutoModelForSequenceClassification

#import sns
import seaborn as sns



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



#### Define Model ####


model_path =  'distilbert-base-multilingual-cased' #'dccuchile/bert-base-spanish-wwm-cased' 
model_name = 'distilbert-multilingual' #'bert-base-spanish' #

# dropout #
config = AutoConfig.from_pretrained(
    model_path,
    num_labels=6,
    hidden_dropout_prob=0.3,  # Default is 0.1; increase for more regularization
    attention_probs_dropout_prob=0.3
)

model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)


# num_epochs = 6
# lr = 0.00005
# batch_size = 12


num_epochs = 20
lr = 0.00001
batch_size = 12

# Define optimizer, loss is already specified as crossEntropyLoss in the model
class_weights = torch.tensor([1.0, 1.0, 1.2, 2.5, 1.8, 3.0]).to(device) #this doesn't get used in the model

optimizer = Adam(model.parameters(), lr=lr)

##add a learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

model.to(device)

#### Train Embeddings ####

#tokenize:
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
full_dataset = tokenize_data(comedias_df, 'tokens', tokenizer)


def train_embeddings(model, train_dataset, num_epochs, batch_size, optimizer, device, pos_weight=None):
    best_epoch = 0
    print("Training model")
    # Initialize a variable to store the highest accuracy
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        
        total_correct = 0
        total_samples = 0
        total_loss = 0.0

        
        batch_count = 1
        for batch in DataLoader(train_dataset, batch_size=batch_size, shuffle=True): # requires_grad=False added: , requires_grad=False
            
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids,attention_mask=attention_mask,labels=labels)

            ####### Calculate loss ########
            loss = outputs.loss

            loss.backward()
            

            optimizer.step()

            # Calculate accuracy
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()


            #print(f'Batch {batch_count} Done!')
            batch_count += 1


        # Calculate epoch statistics
        epoch_accuracy = total_correct / total_samples
        epoch_loss = total_loss / len(train_dataset)

        
        # Print epoch statistics
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2%}')
        
        ##### stop training after 4 epochs if epoch accuracy is below 70 percent
        if epoch == 10 and epoch_accuracy < 0.50:
            print("Epoch accuracy is below 70 percent. Stopping training.")
            break
        
        torch.save(model.state_dict(), 'pca_model.pth')

        #make sure the best model is the one being returned
    model.load_state_dict(torch.load('pca_model.pth'))


    print("Training complete!")

    return model

trained_model = train_embeddings(model, full_dataset, num_epochs, batch_size, optimizer, device, pos_weight=class_weights)


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
embeddings = extract_document_embeddings(trained_model, full_dataset, device)

#### PCA & Plotting ####

archetypes = {v: k for k, v in label_dict.items()}  # Reverse the label_dict
comedias_df['archetype'] = comedias_df['label'].apply(lambda x: archetypes[x])

# Perform PCA for dimensionality reduction to 2D
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

#now plot the embeddings with the archtype labels
comedias_df['archetype'] = comedias_df['label'].apply(lambda x: archetypes[x])
comedias_df['x'] = reduced_embeddings[:, 0]
comedias_df['y'] = reduced_embeddings[:, 1]

# Plot the embeddings
plt.figure(figsize=(10, 10))
sns.scatterplot(data=comedias_df, x='x', y='y', hue='archetype', palette='tab10')
plt.title('Document Embeddings')


# # Add labels to each point
# for i, row in comedias_df.iterrows():
#     plt.annotate(
#         row['character_id'],  # Text label
#         (row['x'], row['y']),  # Position
#         fontsize=8,            # Font size
#         alpha=0.7              # Transparency
#     )


plt.show()




#save the plot to a file
plt.savefig('/archetype-predict-pkg/results/jan22pca.png')





