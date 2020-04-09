import tensorflow as tf
import numpy as np
from src.networks.network import Network
from src.models.supervised import PreUpsamplingModel


class PreUpsamplingNetwork(Network):
    def __init__(self, input_shape):
        model = PreUpsamplingModel(input_shape)
        super().__init__(model)

    def save_state(self):
        pass

    def load_state(self):
        pass

    def train(self, dataset_x, dataset_y, loss_func, epochs, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        with dataset_x:
            with dataset_y:
                for e_idx in range(epochs):
                    # get the data
                    x = next(dataset_x)
                    y_true = next(dataset_y)
                    with tf.GradientTape() as tape:
                        y_pred = self.model(x)  # TODO: switch back
                        loss = loss_func(y_true, y_pred)
                        grads = tape.gradient(loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                    print(f"Epoch: {e_idx} Loss: {np.sum(loss)}")

    def predict(self):
        pass

    def evaluate(self):
        pass
