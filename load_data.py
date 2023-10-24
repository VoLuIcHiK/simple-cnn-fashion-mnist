import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    """Функция вывода картинок на экран,
    :param img: тензор изображения"""
    #unnormalize
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()

def load_data(batch_size):
    """Функция загрузки и подготовки данных,
    :param batch_size: размер пакета данных"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])

    #загрузить картинки из папки, сделать преобразование
    trainset = torchvision.datasets.FashionMNIST(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=2)
    testset = torchvision.datasets.FashionMNIST(root='./data',
                                            train=False,
                                            download=True,
                                            transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=2)
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
    return trainloader, testloader, classes