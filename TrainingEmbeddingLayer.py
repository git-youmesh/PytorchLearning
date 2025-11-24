import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
# Define 10 resturant reviews.
reviews = [
    'Never coming back!',
    'Horrible service',
    'Rude waitress',
    'Cold food.',
    'Horrible food!',
    'Awesome',
    'Awesome service!',
    'Rocks!',
    'poor work',
    'Couldn\'t have done better']

# Define labels (1=negative, 0=positive)
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
VOCAB_SIZE = 50

encoded_reviews = [torch.tensor([hash(word) % VOCAB_SIZE for word in review.split()]) for review in reviews]
MAX_LENGTH = 4
padded_reviews = pad_sequence(encoded_reviews, batch_first=True, padding_value=0).narrow(1, 0, MAX_LENGTH)
print(padded_reviews)
print(encoded_reviews)

model = nn.Sequential(
    nn.Embedding(VOCAB_SIZE, 8),
    nn.Flatten(),
    nn.Linear(8 * MAX_LENGTH, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = optim.Adam(model.parameters())

# Training the model
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(padded_reviews.long())
    loss = criterion(outputs.squeeze(), torch.tensor(labels, dtype=torch.float))
    loss.backward()
    optimizer.step()

embedding_weights = list(model[0].parameters())[0]
print(embedding_weights.shape)
print(embedding_weights)


# Evaluation
with torch.no_grad():
    outputs = model(padded_reviews.long())
    predictions = (outputs > 0.5).float().squeeze()
    accuracy = (predictions == torch.tensor(labels)).float().mean().item()
    loss_value = criterion(outputs.squeeze(), torch.tensor(labels, dtype=torch.float)).item()

print(f'Accuracy: {accuracy}')
print(f'Log-loss: {loss_value}')
