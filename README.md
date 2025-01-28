# archetype-predict-pkg

In this package, we analyze the depiction of the character archetypes galán, dama, rey, reina, criado, and criada, in the works of Pedro Calderón de la Barca. We seek to determine whether the character archetypes are portrayed differently enough from one another, and cohesively enough, for a model to learn these differences and predict the archetype of a character simply based on what they say. We fine-tune a BERT-base model to build embeddings for each character's speech that are sensitive to the character's archetype. We then reduce the dimensionality of the embeddings to visualize how the character archetypes relate to eachother in the embedding space. Finally we use an interpreter to determine which tokens contribute most to the placement of the characters - i.e. which words does a character say that gives them away as being a rey or a criada.

## preprocessing

### tei2csv.py

This file reads plays in tei-xml format and outputs the data into a csv file, where each line is a unique character, including meta data about the character, ex. character archetype, and the play in which they occur. The file is specific to the CalDraCor, but it could be adapted to other plays by changing the desired information.

## functions

### split.py and /split

contains the function to split the model for classification, and outputs the train-val-test split into the /split directory

### test_train.py

a function that outputs the model finetuned on the speech of Calderon's characters.

### pipeline.py

uses the functions in split.py and test_train.py outputs the trained model

### graphing_archetype_pca.ipynb

uses the fine tuned model to create embeddings for all characters, reduces the dimensions of the embeddings to 2, and allows us graph the embedding space. Also allows us to label characters on the graph according to variables like the genre of the play in which they occurred, and their character archetype

### interpret.py

uses the fine-tuned model to interpret embeddings for each character in the corpus based on tokens spoken by the characters.

## analysis

### analyze_archetype_prediction_results.ipynb

taking a closer look at the prediction model, give us the confusion matrix for model predictions

### analyze_interpreter.ipynb

shows the most attributive tokens for each character archetype

## results

the output of the classification task, and graphs are stored here
