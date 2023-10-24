from load_data import load_data, imshow
from train import train
from model import Net
import torchvision
from test import test


if __name__=='__main__':
    batch_size = 4  # размер батча
    PATH = '/content/fmnist_net.pth'
    trainloader, testloader, classes = load_data(batch_size=batch_size)
    data_iter = iter(trainloader)  # создаем итератор
    # кол-во картинок, которые выведутся на экран, равно batch_size
    images, labels = next(data_iter)
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
    model = train(trainloader=trainloader, PATH=PATH)
    test(testloader=testloader, classes=classes, PATH=PATH)



