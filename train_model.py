import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# ===== PARAMETERS =====
IMG_SIZE = (64, 64)
BATCH_SIZE = 8
EPOCHS = 10
STEPS = 5
VAL_STEPS = 2
DATASET_DIR = 'dataset'

# ===== LOAD DATA (Tiny Subset) =====
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# ===== CLASS WEIGHTS (optional) =====
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))

# ===== MODEL =====
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(32, activation='relu')(x)
output = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ===== TRAIN (Tiny Steps) =====
print("🚀 Starting fast training...")
model.fit(
    train_data,
    validation_data=val_data,
    steps_per_epoch=STEPS,
    validation_steps=VAL_STEPS,
    epochs=EPOCHS,
    class_weight=class_weights,
    verbose=1
)

# ===== SAVE MODEL =====
model.save("quick_model.h5")
print("✅ Model saved as quick_model.h5")
