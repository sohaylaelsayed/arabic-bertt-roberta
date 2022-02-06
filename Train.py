import torch.nn as nn
import time
from BertArch import BertArch
import torch
import numpy as np 
from transformers import AdamW
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import StepLR
from torchinfo import summary
from RobertaArch import initialize_roberta_model

class Train():
    device = torch.device("cuda")

    def model(model_name,train_labels,arch):
        # freeze all the parameters. This will prevent updating of model weights during fine-tuning.
        if(arch == 'bert'):
            selected_model = BertArch(model_name)
        elif(arch == 'roberta'):
            selected_model = initialize_roberta_model()
        # push the model to GPU
        selected_model = selected_model.to(Train.device)
        summary(selected_model)
        # define the optimizer
        optimizer = AdamW(selected_model.parameters(), lr = 1e-3)
        #compute the class weights
        class_wts = compute_class_weight(class_weight = "balanced", classes = np.unique(train_labels), y = train_labels)
        print(class_wts)

        #convert class weights to tensor
        weights= torch.tensor(class_wts,dtype=torch.float)
        weights = weights.to(Train.device)
        # loss function
        cross_entropy = nn.NLLLoss(weight=weights) 
        # We can also use learning rate scheduler to achieve better results
        lr_sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        return(selected_model,cross_entropy,optimizer )


    # function to train the model
    def tune_train(bert_model,train_labels,train_dataloader,arch):
        selected_model,cross_entropy,optimizer= Train.model(bert_model,train_labels,arch)
    
        #active training mode
        selected_model.train()
        
        total_loss = 0

        # empty list to save model predictions
        total_preds=[]
            
        # iterate over batches
        for step,batch in enumerate(train_dataloader):
            
            # progress update after every 50 batches.
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step,len(train_dataloader)))
            # push the batch to gpu
            batch = [r.to(Train.device) for r in batch] 
            sent_id, mask, labels = batch
            # get model predictions for the current batch
            preds = selected_model(sent_id, mask)
            # compute the loss between actual and predicted values
            loss = cross_entropy(preds, labels)
            # add on to the total loss
            total_loss = total_loss + loss.item()
            # backward pass to calculate the gradients
            loss.backward()
            # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
            torch.nn.utils.clip_grad_norm_(selected_model.parameters(), 1.0)
            # update parameters
            optimizer.step()
            # clear calculated gradients
            optimizer.zero_grad()  
            # We are not using learning rate scheduler as of now
            # lr_sch.step()
            # model predictions are stored on GPU. So, push it to CPU
            preds=preds.detach().cpu().numpy()
            # append the model predictions
            total_preds.append(preds)
        # compute the training loss of the epoch
        avg_loss = total_loss / len(train_dataloader)
        #accuracy = accuracy_score(train_labels, total_preds) # accuracy of a model
        #print(accuracy) 
        # predictions are in the form of (no. of batches, size of batch, no. of classes).
        # reshape the predictions in form of (number of samples, no. of classes)
        total_preds  = np.concatenate(total_preds, axis=0)
        #returns the loss and predictions
        # empty lists to store training and validation loss of each epoch
        return avg_loss, total_preds,selected_model
            

    def train(bert_model,train_labels,train_dataloader,arch):
        start = time.time()
        train_losses=[]
        # number of training epochs
        epochs = 200
        for epoch in range(epochs):
            # print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
            # print("-"*70) 
            print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
            avg_loss, total_preds,selected_model = Train.tune_train(bert_model,train_labels,train_dataloader,arch)
            #train model
            train_loss, _ = avg_loss, total_preds
            
            # append training and validation loss
            train_losses.append(train_loss)
            # it can make your experiment reproducible, similar to set  random seed to all options where there needs a random seed.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print(f'\nTraining Loss: {train_loss:.3f}')
        stop = time.time()
        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f"Training time: {stop - start}s")
        return(selected_model)

    def save_trained(model_name:str,model_trained):

        if(model_name=="bert"):
            torch.save(model_trained, "model/bert-arb")
        elif(model_name=="roberta"):
           torch.save(model_trained, "model/roberta-arb ")
        elif(model_name=="distilbert"):
            torch.save(model_trained, "model/distilbert")
        return()