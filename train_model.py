import numpy as np
from sklearn.svm import SVC
import pickle

# Create synthetic training data
np.random.seed(42)
X_train = np.random.randn(200, 7)
y_train = np.random.randint(0, 2, 200)

# Train a simple SVC model without probability
model = SVC(kernel='rbf', probability=False)
model.fit(X_train, y_train)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print('Model trained and saved successfully!')
