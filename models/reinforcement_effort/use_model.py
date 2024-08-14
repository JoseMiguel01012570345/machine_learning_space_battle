from keras.models import load_model
import gym

model_path = './saved_models/my_model_folder/model.h5'

loaded_model = load_model(model_path)

# Make a prediction
predictions = loaded_model.predict(dummy_input_data)

print(predictions)