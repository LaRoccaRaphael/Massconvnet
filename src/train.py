import os
import time
import math
import torch
import logging
import numpy as np
from tqdm import tqdm
from evaluation import AverageMeter, ConfusionMatrix

def trainer(train_loader,
            val_loader,
            test_loader,
            model,
            optimizer,
            scheduler,
            criterion,
            model_name,
            max_epochs=1000,
            evaluation_frequency=20):

    logging.info("start training")

    best_loss = 9e99
    best_metric = -1

    for epoch in range(max_epochs):
        best_model_path = os.path.join("models", model_name, "model.pth.tar")


        # train for one epoch
        loss_training, performance_training = train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch + 1,
            train = True)

        # evaluate on validation set
        with torch.no_grad():
            loss_validation, performance_validation = train(
                val_loader,
                model,
                criterion,
                optimizer,
                epoch + 1,
                train = False)

        

        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }
        os.makedirs(os.path.join("models", model_name), exist_ok=True)

        # Remember best loss and save checkpoint
        #is_better = loss_validation < best_loss
        #best_loss = min(loss_validation, best_loss)
        is_better = performance_validation["accuracy"] > best_metric
        best_metric = max(performance_validation["accuracy"], best_metric)

        # Save the best model based on loss
        if is_better:
            logging.info("Validation performance at epoch " + str(epoch+1) + " -> " + str(performance_validation["accuracy"]))
        
        # Compute the performance on the testset
        if (epoch+1)%evaluation_frequency == 0:     
            performance_test = test(
                        test_loader,
                        model, 
                        model_name)
            logging.info("Test performance at epoch " + str(epoch+1) + " -> " + str(performance_test["accuracy"]))
            torch.save(state, best_model_path)
             
        # Learning rate scheduler update
        prevLR = optimizer.param_groups[0]['lr']
        scheduler.step(loss_validation)
        currLR = optimizer.param_groups[0]['lr']
        if (currLR is not prevLR and scheduler.num_bad_epochs == 0):
            logging.info("Plateau Reached!")
        if (prevLR < 2 * scheduler.eps and
                scheduler.num_bad_epochs >= scheduler.patience):
            logging.info(
                "Plateau Reached and no more reduction -> Exiting Loop")
            break

    return

def train(dataloader,
          model,
          criterion,
          optimizer,
          epoch,
          train=False):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    confusion_matrix = ConfusionMatrix(dataloader.dataset.num_classes)

    # Switch to train mode
    if train:
        model.train()
    else:
        model.eval()
        
    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=160) as t:
        for i, (data) in t: 
            # measure data loading time
            data_time.update(time.time() - end)

            data = data.cuda()

            # compute output
            output = model(data)

            loss = criterion(output,data.y)
            confusion_matrix.update(output,data.y)

            # measure accuracy and record loss
            losses.update(loss.item(), data.y.size()[0])

            if train:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if train:
                desc = f'Train {epoch}: '
            else:
                desc = f'Evaluate {epoch}: '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            desc += f'Loss {losses.avg:.4e} '
            t.set_description(desc)

    print(confusion_matrix)
    return losses.avg, confusion_matrix.performance()




def test(dataloader,model, model_name):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()


    confusion_matrix = ConfusionMatrix(dataloader.dataset.num_classes)

    model.eval()

    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120) as t:
        for i, (data) in t:
            data_time.update(time.time() - end)

            data = data.cuda()

            output = model(data)

            confusion_matrix.update(output,data.y)
    print(confusion_matrix)
    return confusion_matrix.performance()


def CAM(dataloader,model):

    model.eval()
        
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=160) as t:
        for i, (data) in t: 
            # measure data loading time

            data = data.cuda()

            # compute output
            output = model.CAM(data)