from model import Net
import torch


def accuracy(net, dataloader):
    """Функия расчета accuracy,
    :param net: загруженная модель,
    :param dataloader: загрузчик тестовых даннных"""
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the {total} test images: {100 * correct / total} %')
    return 100 * correct / total



def test(testloader, PATH):
    """Функция тестирования обученной сети,
    :param testloader: даталоэдер тестивоых данных,
    :param PATH: путь до весов обученной модели"""
    net = Net()
    net.load_state_dict(torch.load(PATH))
    net.eval()
    return accuracy(net, testloader)
