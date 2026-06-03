import os
import tensorflow as tf
from tensorflow.keras import layers, models

print("Starting model generation...")

# 1. Create the 'models' directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# 2. Build a standard Neural Network placeholder
# (Assuming standard 48x48 grayscale images for mood analysis like FER2013)
model = models.Sequential([
    layers.Input(shape=(48, 48, 1)), 
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(7, activation='softmax') # 7 standard emotions
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Save it directly to the path your app is looking for
model.save('models/mood_model.keras')

print("✨ Success! 'models/mood_model.keras' has been created.")