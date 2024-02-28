import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# PREPROCESSING
# clean data
clean_lines = []
with open('urlset.csv', 'rb') as file:  # Open in binary mode
    for line in file:
        try:
            clean_lines.append(line.decode('utf-8'))  # Try to decode each line
        except UnicodeDecodeError:
            pass  # Skip lines that cause decoding errors

with open('cleaned_urlset.csv', 'w', encoding='utf-8') as clean_file:
    clean_file.writelines(clean_lines)

# Step 1: Load the dataset
df = pd.read_csv('cleaned_urlset.csv')  

# Remove unnecessary columns
df = df[['domain', 'label']]

#Preprocess the data further
def extract_features(url):
    num_slashes = url.count('/')
    num_dots = url.count('.')
    return [num_slashes, num_dots]

# Apply feature extraction to each URL
features = df['domain'].apply(extract_features)

# Convert features and labels into a tensor
features_tensor = torch.tensor(features.tolist(), dtype=torch.float32)
labels_tensor = torch.tensor(df['label'].values, dtype=torch.float32)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features_tensor, labels_tensor, test_size=0.2, random_state=42)

# DataLoader
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Neural Network Model
class URLClassifier(nn.Module):
    def __init__(self):
        super(URLClassifier, self).__init__()
        self.fc1 = nn.Linear(2, 5)  # Adjust according to the number of features
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

model = URLClassifier()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(10):  # Number of epochs
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(torch.float32), labels.to(torch.float32)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Make predictions on the test set
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation
    predictions = model(X_test).squeeze()  # Squeeze to remove extra dimensions
    predictions = torch.round(predictions)  # Round to get binary predictions

# Convert predictions and true labels to CPU numpy arrays for evaluation with sklearn
predictions = predictions.cpu().numpy()
y_test_np = y_test.cpu().numpy()

# Calculate metrics
accuracy = accuracy_score(y_test_np, predictions)
precision = precision_score(y_test_np, predictions)
recall = recall_score(y_test_np, predictions)
f1 = f1_score(y_test_np, predictions)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')