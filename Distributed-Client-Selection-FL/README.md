# Federated Learning

This project is the implementation of paper, "Communication-Loss Trade-Off in Federated Learning: A Distributed Client Selection Algorithm".

Abstract:
Mass data generation occurring in the Internet-of-Things (IoT) requires processing to extract meaningful information. Deep learning is commonly used to perform such processing. However, due to the sensitive nature of these data, it is important to consider data privacy. As such, federated learning (FL) has been proposed to address this issue. FL pushes training to the client devices and tasks a central server with aggregating collected model weights to update a global model. However, the transmission of these model weights can be costly, gradually. The trade-off between communicating model weights for aggregation and the loss provided by the global model remains an open problem. In this work, we cast this trade-off problem of client selection in FL as an optimization problem. We then design a Distributed Client Selection (DCS) algorithm that allows client devices to decide to participate in aggregation in hopes of minimizing overall communication cost --- while maintaining low loss. We evaluate the performance of our proposed client selection algorithm against standard FL and a state-of-the-art client selection algorithm, called Power-of-Choice (PoC), using CIFAR-10, FMNIST, and MNIST datasets. Our experimental results confirm that our DCS algorithm is able to closely match the loss provided by the standard FL and PoC, while on average reducing the overall communication cost by nearly 32.67% and 44.71% in comparison to standard FL and PoC, respectively.

## Ackonwledgements
Acknowledgements give to [shaoxiongji](https://github.com/shaoxiongji/federated-learning).


To run, use this command:
python test.py --dataset mnist --num_users 100 --num_channels 1 --model cnn --epochs 50 --gpu -1 --lr 0.001

for cifar10, use the channel number=3
