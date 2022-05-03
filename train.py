import math
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from model import CNNmodel, DNNmodel, fp_posmodel
from utils import binary_acc
import torch
import torch.nn as nn
from config import titanic_config as config
from datapre import get_loader
import torch.nn.functional as F


device = config.device

def train(model, train_loader, test_loader, config, criterion, optimizer, scheduler):
    #logger = logger_configuration(config, save_log=True)

    #data_len = len(train_loader)

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            ##summary(model, spectrograms)   #显示参数
            y_pred = model(X_batch)  #
            loss = criterion(y_pred, y_batch)
            acc = binary_acc(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            if scheduler != None:
                scheduler.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        print(
            f'Epoch {epoch + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')

        model.eval()
        if (epoch+1) % config.test_step == 0:
            test(model, test_loader, criterion)
        if (epoch+1) % config.save_step == 0:
            torch.save(model.state_dict(), '/Ep{}.pth'.format(epoch+1))


def test(model, test_loader, criterion):
    print('\nevaluating…')
    model.eval()

    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)  #
            loss = criterion(y_pred, y_batch)
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        print(
            f'Loss: {epoch_loss / len(test_loader):.5f} | Acc: {epoch_acc / len(test_loader):.3f}')

if __name__ == "__main__":
    train_loader, test_loader = get_loader()
    model = DNNmodel()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = None
    train(model, train_loader, test_loader, config, criterion, optimizer, scheduler)
