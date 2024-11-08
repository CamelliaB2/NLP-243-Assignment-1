import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

#Read training csv file
df = pd.read_csv('hw1_train.csv')

#Showcase first 5 rows of training set
df.head()

df['CORE RELATIONS'] = df['CORE RELATIONS'].fillna('') #Assign NaN values as an empty string
df['CORE RELATIONS'] = df['CORE RELATIONS'].apply(lambda x: x.split() if isinstance(x, str) else {}) #Split string into list of words
x = df['UTTERANCES'] #Assign utterances as x
y = df['CORE RELATIONS'] #Assign core relations as y

mlb = MultiLabelBinarizer() #Converts to binary format

#Char-level tokenization
vectorizer = CountVectorizer(
    analyzer='char_wb',          
    ngram_range=(1, 3),   
    max_features=5000,        
    min_df=2                  
)

x_train = vectorizer.fit_transform(x).toarray() #Assign training data for x
y_train = mlb.fit_transform(y) #Assign training data for y

#Data split, 80% of data for training and 20% for testing
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

#Convert to tensors for training and testing
x_train_vec = torch.FloatTensor(x_train)
x_test_vec = torch.FloatTensor(x_test)
y_train_vec = torch.FloatTensor(y_train)
y_test_vec = torch.FloatTensor(y_test)

#Normalization
mean, std = x_train_vec.mean(), x_train_vec.std()
x_train_vec = (x_train_vec - mean) / std
x_test_vec = (x_test_vec - mean) / std

#Define model

class SentimentClassifier(nn.Module):
    def __init__(self, input_dim, output_data):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512) 
        self.fc2 = nn.Linear(512, 256) #First layer with 512 neurons
        self.fc3 = nn.Linear(256, output_data) #Second layer with 256 neurons
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):  # x (batch_size, input_dim)
        x = self.leaky_relu(self.fc1(x))  # (batch_size, 256)
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))  # (batch_size, 64)
        x = self.dropout(x)
        x = self.fc3(x)  # (batch_size, 1)
        x = x.squeeze()
        return x  # (batch_size,)

#Store input data as x and outpur data y
class ClassificationDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
train_dataset = ClassificationDataset(x_train_vec, y_train_vec) #Store training dataset using Classification Dataset class
test_dataset = ClassificationDataset(x_test_vec, y_test_vec) #Store testing dataset using Classification Dataset class
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) #Load training data with batch size of 64
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False) #Load testing data with batch size of 64

# initialize the model
model = SentimentClassifier(x_train_vec.size(1), len(mlb.classes_))

# define loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

#Code adapted from Sicong Huang from sections #1 and #2

num_epoch = 100
accuracies = []
losses = []

# training loop
for epoch in range(num_epoch):

    running_loss = torch.tensor(0.)
    for x_batch, y_batch in train_loader:


        #forward pass
        model.train()

        optimizer.zero_grad()
        output = model(x_batch)
        loss = loss_fn(output, y_batch)

        running_loss += loss.detach().item()

        #backward pass
        loss.backward()
        optimizer.step()

    avg_loss = running_loss / len(train_loader)  # Ensure you average over the number of batches
    losses.append(avg_loss) 

    # evaluate on test set
    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            output = model(x_batch)
            predicted = (torch.sigmoid(output) > 0.5).float()
            correct_predictions += (predicted == y_batch).all(dim=1).sum().item()
            total_samples += y_batch.size(0)

        acc = correct_predictions / total_samples
        accuracies.append(acc * 100)
        print(f'Epoch {epoch + 1}/{num_epoch}, Loss: {avg_loss:.4f}, Acc: {acc * 100:.2f}')

fig, ax1 = plt.subplots()

# Plot accuracy on the primary y-axis
ax1.plot(range(1, num_epoch + 1), accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (%)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Create a second y-axis for the loss
ax2 = ax1.twinx()
ax2.plot(range(1, num_epoch + 1), losses, marker='x', linestyle='-', color='r', label='Loss')
ax2.set_ylabel('Loss', color='r')
ax2.tick_params(axis='y', labelcolor='r')

plt.title('Accuracy and Loss over Epochs')
fig.tight_layout()
plt.grid(True)
plt.show()

# Load the test data
df_test = pd.read_csv('hw1_test.csv')
x = df_test['UTTERANCES']
x_test_final = vectorizer.transform(x).toarray()
x_test_final_vec = torch.FloatTensor(x_test_final)


model.eval()

predictions = []

with torch.no_grad():
    outputs = model(x_test_final_vec)
    predicted = (torch.sigmoid(outputs) > 0.5).float()

core_relations = mlb.inverse_transform(predicted.cpu().numpy())
submission_df = pd.DataFrame({
    'ID' : df_test['ID'],
    'CORE RELATIONS' : ['none' if len(r) == 0 else ' '.join(r) for r in core_relations]
})

submission_df.to_csv('predictions_submission.csv', index=False)

print(f"Predictions saved in the desired format")