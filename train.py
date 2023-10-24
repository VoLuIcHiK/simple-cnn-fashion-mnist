import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from model import Net


def train(trainloader, PATH):
    """Функция обучения нейронной сети,
    :param trainloader: даталоэдер тренировочных данных,
    :param PATH: путь к файлу, где будут сохранены веса"""
    wandb.login()
    run = wandb.init(project="fashion_mnist", reinit=True)
    config = run.config
    config.learning_rate = 0.0001
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    for epoch in range(20):
        running_loss = 0.01
        for i, data in enumerate(trainloader):
            #список [images, labels]
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
                run.log({"loss": loss})

    print('Finished Training')
    # save learned model
    torch.save(model.state_dict(), PATH)
    run.finish()
    return model
