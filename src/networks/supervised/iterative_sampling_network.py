import tensorflow as tf

from src.config import *
from src.networks.network import Network
from src.models import IterativeSamplingModel


class IterativeSamplingNetwork(Network):

    def __init__(self):
        model = IterativeSamplingModel(scaling_factor=2)
        super().__init__(model)

    @tf.function
    def train_step(self, x, y, optimizer, loss_func):
        with tf.GradientTape() as tape:
            y_pred = self.model(x)
            loss = loss_func(y, y_pred)
            grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return tf.reduce_sum(loss)

    @tf.function
    def valid_step(self, x, y, loss_func):
        y_pred = self.model(x)
        loss = loss_func(y, y_pred)
        return tf.reduce_sum(loss)

    def train(self, dataset_x, dataset_y, loss_func, epochs, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        train_x, valid_x = dataset_x.split((8, 2))
        train_y, valid_y = dataset_y.split((8, 2))
        with train_x, train_y, valid_x, valid_y:
            for e_idx in range(epochs):
                train_loss = 0
                for x, y in zip(train_x, train_y):
                    train_loss += self.train_step(x, y, optimizer, loss_func)
                valid_loss = 0
                for x, y in zip(valid_x, valid_y):
                    valid_loss += self.valid_step(x, y, loss_func)
                LOGGER.info(f"Epoch: {e_idx} train_loss: {train_loss:.2f} valid_loss: {valid_loss:.2f}")
                if e_idx > 0 and e_idx % 100 == 0:
                    LOGGER.info(f"Saving state at {e_idx + 1} epochs.")
                    self.save_state()
