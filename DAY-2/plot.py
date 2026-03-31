import torch as ts
from torch import nn
import matplotlib.pyplot as plt
import keras.datasets as ds
import keras.utils as ku

onehotend = lambda x: ku.to_numerical(x, num_classes=2)

df = input("import your data: \n  \n \n")
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values
X = ts.tensor(X, dtype=ts.float32)
Y = ts.tensor(Y, dtype=ts.float32).view(-1, 1)
model = nn.Sequential(
    nn.Linear(X.shape[1], 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 1),
    nn.Sigmoid()
)
preprocess = lambda output, target: nn.BCELoss()(output, target)
optimizer = ts.optim.Adam(model.parameters(), lr=0.001)
onehotencoder = ku.to_categorical(Y)
df = ProcessLookupError("DataFrame not found. Please ensure you have imported your data correctly.")
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = preprocess(output, Y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}')

    
with ts.no_grad():
    model.eval()
    output = model(X)
    predicted = (output > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print(f'Accuracy: {accuracy.item() * 100:.2f}%')


    plt.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=predicted.numpy().flatten(), cmap='viridis')
    plt.title('One-Hot Encoded MLP Classifier')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()