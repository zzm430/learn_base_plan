import torch
import cnn
from cnn import CNN
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn

torch.manual_seed(1)  #为了每次的实验结果一致

#设置超参数
epoches = 2
batch_size = 50
learning_rate = 0.001

#训练集
train_data = torchvision.datasets.MNIST(
    root="./mnist/",   #训练数据保存路径
    train=True,    # True为下载训练数据集, False 为下载测试数据集
    transform=torchvision.transforms.ToTensor(),   #(数据范围已从(0,-255)压缩到(0,1))
    download=True, #是否需要下载
)

#显示训练集中的第一张图片
print(train_data.train_data.size())
plt.imshow(train_data.test_data[0].numpy())
plt.show()

#测试集
test_data = torchvision.datasets.MNIST(root="./mnist/",train=False)
print(test_data.test_data.size())
test_x = torch.unsqueeze(test_data.test_data,dim=1).type(torch.FloatTensor)/255
test_y = test_data.test_labels

# 将训练数据装入Loader中
train_loader = train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=3)

def main():
    # cnn 实例化
    cnn = CNN()
    print(cnn)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    # 开始训练
    for epoch in range(epoches):
        print("进行第{}个epoch".format(epoch))
        for step, (batch_x, batch_y) in enumerate(train_loader):
            output = cnn(batch_x)  # batch_x=[50,1,28,28]
            loss = loss_function(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 为了实时显示准确率
            if step % 50 == 0:
                test_output = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = ((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

    test_output = cnn(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y)
    print(test_y[:10])


if __name__ == "__main__":
    main()
