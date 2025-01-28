import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

def test_model(model, test_dataset, batch_size, device):
    print("Testing model")
    # Evaluation loop
    
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        model.eval()
        all_predictions = []
        all_probabilities = []
        for batch in DataLoader(test_dataset, batch_size=batch_size):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)


            probabilities = torch.nn.functional.softmax(logits, dim=1)

            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    accuracy = total_correct / total_samples
    print("Testing Complete!")
    return accuracy, all_predictions, all_probabilities


def train_model(model, train_dataset, val_dataset, num_epochs, batch_size, optimizer, device, pos_weight=None):
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

        ###############################################################################################################################
        
        #after 4 epochs, save the best model and stop training once the accuracy starts to decrease
        if epoch > 5:
            print("Testing model on validation data...")
        ## test on dev, save model from & record which epoch we stopped on and keep the model which gives the highest accuracy ########

            # Evaluate the model on the test dataset after every epoch
            accuracy, all_predictions, all_probabilities = test_model(model, val_dataset, batch_size, device)
            print(f'Accuracy on validation data: {accuracy:.2%}')

            ###################################
            #why is early stopping not working?
            if best_accuracy - accuracy > .05:
                print("Accuracy going down. Stopping training.")
                break
            #################################

            # Save the model if it achieves the highest accuracy on the test dataset so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy

                best_epoch = epoch
                # Save the model here
                torch.save(model.state_dict(), 'best_model.pth')

        # print("True labels:", labels)
        # print("Predictions:", predictions)  

        #make sure the best model is the one being returned
    model.load_state_dict(torch.load('best_model.pth'))


        ########################################################################################################
        # Evaluate the model on the test dataset


    print("Training complete!")

    return model, best_epoch