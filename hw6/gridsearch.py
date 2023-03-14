from classification import *
from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

def train_cifar( config,checkpoint_dir =None, mymodel=None, train_dataloader=None, validation_dataloader=None, device=None):
    """ Train a PyTorch Module

    :param torch.nn.Module mymodel: the model to be trained
    :param int num_epochs: number of epochs to train for
    :param torch.utils.data.DataLoader train_dataloader: DataLoader containing training examples
    :param torch.utils.data.DataLoader validation_dataloader: DataLoader containing validation examples
    :param torch.device device: the device that we'll be training on
    :param float lr: learning rate
    :return None
    """
    lr  = config["lr"]
    #num_epochs = config["num_epochs"]
    # here, we use the AdamW optimizer. Use torch.optim.Adam.
    # instantiate it on the untrained model parameters with a learning rate of 5e-5
    print(" >>>>>>>>  Initializing optimizer")
    optimizer = torch.optim.AdamW(mymodel.parameters(), lr=lr)

    # now, we set up the learning rate scheduler
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * num_epochs
    )

    loss = torch.nn.CrossEntropyLoss()

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train_acc=[]
    val_acc=[]
    for epoch in range(9):

        # put the model in training mode (important that this is done each epoch,
        # since we put the model into eval mode during validation)
        mymodel.train()

        # load metrics
        train_accuracy = evaluate.load('accuracy')

        print(f"Epoch {epoch + 1} training:")

        for i, batch in enumerate(train_dataloader):

            """
            You need to make some changes here to make this function work.
            Specifically, you need to: 
            Extract the input_ids, attention_mask, and labels from the batch; then send them to the device. 
            Then, pass the input_ids and attention_mask to the model to get the logits.
            Then, compute the loss using the logits and the labels.
            Then, call loss.backward() to compute the gradients.
            Then, call optimizer.step()  to update the model parameters.
            Then, call lr_scheduler.step() to update the learning rate.
            Then, call optimizer.zero_grad() to reset the gradients for the next iteration.
            Then, compute the accuracy using the logits and the labels.
            """

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            output = mymodel(input_ids=input_ids,attention_mask=attention_mask) #logits
            predictions = output.logits
            model_loss = loss(predictions, batch['labels'].to(device))

            model_loss.backward()
            optimizer.step()

            lr_scheduler.step()
            optimizer.zero_grad()

            predictions = torch.argmax(predictions, dim=1)

            # update metrics
            train_accuracy.add_batch(predictions=predictions, references=batch['labels'])

        # print evaluation metrics
        curr_train_acc = train_accuracy.compute()
        train_acc.append(curr_train_acc['accuracy'])
        print(f" ===> Epoch {epoch + 1}")
        print(f" - Average training metrics: accuracy={curr_train_acc}")

        # normally, validation would be more useful when training for many epochs
        val_accuracy = evaluate_model(mymodel, validation_dataloader, device)
        val_acc.append(val_accuracy['accuracy'])
        print(f" - Average validation metrics: accuracy={val_accuracy}")

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((mymodel.state_dict(), optimizer.state_dict()), path)

        tune.report(accuracy=val_accuracy['accuracy'])
        print("Finished Training")
    #return train_acc,val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--small_subset", action='store_true')
    #parser.add_argument("--num_epochs", type=int, default=1)
    #parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")

    args = parser.parse_args()
    print(f"Specified arguments: {args}")

    assert type(args.small_subset) == bool, "small_subset must be a boolean"

    print("the value of small_subset: ", args.small_subset)

    config = {
        "lr": tune.grid_search([1e-4, 5e-4, 1e-3])
        #"num_epochs": tune.choice([5,7,9])
    }
    gpus_per_trial = 1
    # load the data and models
    pretrained_model, train_dataloader, validation_dataloader, test_dataloader = pre_process(args.model,
                                                                                             args.batch_size,
                                                                                             args.device,
                                                                                             args.small_subset)

    print(" >>>>>>>>  Starting training ... ")
    #train_acc, val_acc = train(pretrained_model, args.num_epochs, train_dataloader, validation_dataloader, args.device,
    #                           args.lr)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["accuracy", "training_iteration"])
    result = tune.run(
        partial(train_cifar, mymodel=pretrained_model, train_dataloader=train_dataloader, validation_dataloader=validation_dataloader, device=args.device ),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=config,
        #num_samples=num_samples,
        #scheduler=scheduler,
        progress_reporter=reporter)
    # print the GPU memory usage just to make sure things are alright
    print_gpu_memory()

    best_trial = result.get_best_trial("accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    pretrained_model.load_state_dict(model_state)

    val_accuracy = evaluate_model(pretrained_model, validation_dataloader, args.device)
    print(f" - Average DEV metrics: accuracy={val_accuracy}")

    test_accuracy = evaluate_model(pretrained_model, test_dataloader, args.device)
    print(f" - Average TEST metrics: accuracy={test_accuracy}")