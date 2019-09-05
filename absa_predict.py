from keras.models import load_model
atae_model = load_model("atae_model.h5")
atae_model.evaluate()
