import os
import json
from django.conf import settings
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, preprocessing, applications

# ----------------------------------------------------------------cnn model training-----------------------------------------
def train_model():
    try:
        # Define dataset and model paths
        dataset_dir = os.path.join(settings.MEDIA_ROOT, 'augmented_dataset')
        model_save_path = os.path.join(settings.MEDIA_ROOT, 'models', 'best_cnn_model.h5')

        # Data generators for training and validation
        datagen = preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

        train_gen = datagen.flow_from_directory(
            dataset_dir, target_size=(299, 299), batch_size=32,
            class_mode='sparse', subset='training', shuffle=True)

        val_gen = datagen.flow_from_directory(
            dataset_dir, target_size=(299, 299), batch_size=32,
            class_mode='sparse', subset='validation', shuffle=True)

        # Model definition with Input layer (resolves input_shape warning)
        model = models.Sequential([
            layers.Input(shape=(299, 299, 3)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(2, activation='softmax')
        ])

        # Compile model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Save only weights without optimizer state (avoid optimizer warning)
        checkpoint = callbacks.ModelCheckpoint(
            model_save_path, monitor='val_accuracy',
            save_best_only=True, mode='max', verbose=1,
            save_weights_only=False
        )

        # Train model
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=13,
            callbacks=[checkpoint]
        )

        # Save model without optimizer state to avoid optimizer-related warning on load
        model.save(model_save_path, include_optimizer=False)

        # Get final metrics
        final_val_acc = history.history['val_accuracy'][-1]
        final_val_loss = history.history['val_loss'][-1]

        return {
            'success': True,
            'message': "Model training completed successfully.",
            'val_accuracy': round(final_val_acc * 100, 2),
            'val_loss': round(final_val_loss, 4)
        }

    except Exception as e:
        return {
            'success': False,
            'message': f"Training failed: {str(e)}",
            'val_accuracy': None,
            'val_loss': None
        }

# ----------------------------------------------------------------cnn json model training-----------------------------------------

def json_train_model():
    try:
        dataset_dir = os.path.join(settings.MEDIA_ROOT, 'augmented_dataset')
        model_save_path = os.path.join(settings.MEDIA_ROOT, 'models', 'best_cnn_model.h5')
        result_json_path = os.path.join(settings.MEDIA_ROOT, 'models', 'cnn_training_result.json')

        # Step 1: If result JSON exists, return the cached results
        if os.path.exists(result_json_path):
            with open(result_json_path, 'r') as f:
                cached_result = json.load(f)
            cached_result['message'] = "Loaded cached training results."
            return cached_result

        # Step 2: If not cached, proceed with training
        datagen = preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

        train_gen = datagen.flow_from_directory(
            dataset_dir, target_size=(299, 299), batch_size=32,
            class_mode='sparse', subset='training', shuffle=True)

        val_gen = datagen.flow_from_directory(
            dataset_dir, target_size=(299, 299), batch_size=32,
            class_mode='sparse', subset='validation', shuffle=True)

        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(299, 299, 3)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(2, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        checkpoint = callbacks.ModelCheckpoint(
            model_save_path, monitor='val_accuracy',
            save_best_only=True, mode='max', verbose=1)

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=13,
            callbacks=[checkpoint]
        )

        final_val_acc = round(history.history['val_accuracy'][-1] * 100, 2)
        final_val_loss = round(history.history['val_loss'][-1], 4)

        result = {
            'success': True,
            'message': "Model training completed successfully.",
            'val_accuracy': final_val_acc,
            'val_loss': final_val_loss
        }

        # Step 3: Save result to JSON file
        with open(result_json_path, 'w') as f:
            json.dump(result, f)

        return result

    except Exception as e:
        return {
            'success': False,
            'message': f"Training failed: {str(e)}",
            'val_accuracy': None,
            'val_loss': None
        }


# ------------------------------------------------------------------resenet model training----------------------------------------------------


def resnet_model():
    try:
        # Paths
        dataset_dir = os.path.join(settings.MEDIA_ROOT, 'augmented_dataset')
        model_save_path = os.path.join(settings.MEDIA_ROOT, 'models', 'best_resnet_model.h5')

        # Data generators
        datagen = preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

        train_gen = datagen.flow_from_directory(
            dataset_dir, target_size=(224, 224), batch_size=32,
            class_mode='sparse', subset='training', shuffle=True)

        val_gen = datagen.flow_from_directory(
            dataset_dir, target_size=(224, 224), batch_size=32,
            class_mode='sparse', subset='validation', shuffle=True)

        # Load base ResNet50 model
        base_model = applications.ResNet50(weights='imagenet', include_top=False, input_tensor=layers.Input(shape=(224, 224, 3)))
        base_model.trainable = False  # Freeze base layers

        # Add custom top layers
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation='relu')(x)
        predictions = layers.Dense(2, activation='softmax')(x)

        model = models.Model(inputs=base_model.input, outputs=predictions)

        model.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Model checkpoint
        checkpoint = callbacks.ModelCheckpoint(
            model_save_path, monitor='val_accuracy',
            save_best_only=True, mode='max', verbose=1,
            save_weights_only=False
        )

        # Train the model
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=13,
            callbacks=[checkpoint]
        )

        # Save model without optimizer to avoid warnings
        model.save(model_save_path, include_optimizer=False)

        final_val_acc = history.history['val_accuracy'][-1]
        final_val_loss = history.history['val_loss'][-1]

        return {
            'success': True,
            'message': "ResNet model training completed successfully.",
            'val_accuracy': round(final_val_acc * 100, 2),
            'val_loss': round(final_val_loss, 4)
        }

    except Exception as e:
        return {
            'success': False,
            'message': f"Training failed: {str(e)}",
            'val_accuracy': None,
            'val_loss': None
        }

# ------------------------------------------------------------------resnet json model training----------------------------------------------------

def json_resnet_model():
    try:
        # File paths
        dataset_dir = os.path.join(settings.MEDIA_ROOT, 'augmented_dataset')
        model_dir = os.path.join(settings.MEDIA_ROOT, 'models')
        os.makedirs(model_dir, exist_ok=True)

        model_save_path = os.path.join(model_dir, 'best_resnet_model.h5')
        result_json_path = os.path.join(model_dir, 'resnet_result.json')

        # If result JSON exists, load and return immediately
        if os.path.exists(result_json_path):
            with open(result_json_path, 'r') as json_file:
                cached_result = json.load(json_file)
            cached_result['message'] = "Loaded cached ResNet result."
            return cached_result

        # Data generators
        datagen = preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

        train_gen = datagen.flow_from_directory(
            dataset_dir, target_size=(224, 224), batch_size=32,
            class_mode='sparse', subset='training', shuffle=True)

        val_gen = datagen.flow_from_directory(
            dataset_dir, target_size=(224, 224), batch_size=32,
            class_mode='sparse', subset='validation', shuffle=True)

        # Build ResNet50 model
        base_model = applications.ResNet50(weights='imagenet', include_top=False, input_tensor=layers.Input(shape=(224, 224, 3)))
        base_model.trainable = False

        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation='relu')(x)
        predictions = layers.Dense(2, activation='softmax')(x)

        model = models.Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        checkpoint = callbacks.ModelCheckpoint(
            model_save_path, monitor='val_accuracy', save_best_only=True,
            mode='max', verbose=1, save_weights_only=False
        )

        # Train
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=13,
            callbacks=[checkpoint]
        )

        # Save model without optimizer
        model.save(model_save_path, include_optimizer=False)

        # Prepare result
        final_val_acc = history.history['val_accuracy'][-1]
        final_val_loss = history.history['val_loss'][-1]

        result = {
            'success': True,
            'message': "ResNet model trained and result cached successfully.",
            'val_accuracy': round(final_val_acc * 100, 2),
            'val_loss': round(final_val_loss, 4)
        }

        # Save result to JSON for caching
        with open(result_json_path, 'w') as f:
            json.dump(result, f)

        return result

    except Exception as e:
        return {
            'success': False,
            'message': f"Training failed: {str(e)}",
            'val_accuracy': None,
            'val_loss': None
        }

# ------------------------------------------------------------------xception model training----------------------------------------------------

def xception_model():
    try:
        # Paths
        dataset_dir = os.path.join(settings.MEDIA_ROOT, 'augmented_dataset')
        model_dir = os.path.join(settings.MEDIA_ROOT, 'models')
        os.makedirs(model_dir, exist_ok=True)

        model_save_path = os.path.join(model_dir, 'best_xception_model.h5')
        json_result_path = os.path.join(model_dir, 'xception_model_result.json')

        # Data generators
        datagen = preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

        train_gen = datagen.flow_from_directory(
            dataset_dir, target_size=(299, 299), batch_size=32,
            class_mode='sparse', subset='training', shuffle=True)

        val_gen = datagen.flow_from_directory(
            dataset_dir, target_size=(299, 299), batch_size=32,
            class_mode='sparse', subset='validation', shuffle=True)

        # Load base Xception model
        base_model = applications.Xception(weights='imagenet', include_top=False, input_tensor=layers.Input(shape=(299, 299, 3)))
        base_model.trainable = False  # Freeze base layers

        # Add custom top layers
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation='relu')(x)
        predictions = layers.Dense(2, activation='softmax')(x)

        model = models.Model(inputs=base_model.input, outputs=predictions)

        model.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Model checkpoint
        checkpoint = callbacks.ModelCheckpoint(
            model_save_path, monitor='val_accuracy',
            save_best_only=True, mode='max', verbose=1,
            save_weights_only=False
        )

        # Train the model
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=13,
            callbacks=[checkpoint]
        )

        # Save model without optimizer to avoid warnings
        model.save(model_save_path, include_optimizer=False)

        final_val_acc = history.history['val_accuracy'][-1]
        final_val_loss = history.history['val_loss'][-1]

        result = {
            'success': True,
            'message': "Xception model training completed successfully.",
            'val_accuracy': round(final_val_acc * 100, 2),
            'val_loss': round(final_val_loss, 4)
        }

        # Save result as JSON for fast reloading later
        with open(json_result_path, 'w') as json_file:
            json.dump(result, json_file)

        return result

    except Exception as e:
        return {
            'success': False,
            'message': f"Training failed: {str(e)}",
            'val_accuracy': None,
            'val_loss': None
        }
