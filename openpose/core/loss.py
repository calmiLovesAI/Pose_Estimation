import tensorflow as tf


class OpenPoseLoss:
    def __init__(self):
        self.weights = [5, 5, 5, 5, 1, 1]

    def __call__(self, y_true, y_pred, *args, **kwargs):
        assert len(y_true) == len(y_pred)
        n = len(y_true)
        total_loss = 0.0
        for i in range(n):
            total_loss += self.weights[i] * OpenPoseLoss.__single_output_loss(y_true[i], y_pred[i])
        return total_loss

    @staticmethod
    def __single_output_loss(y_true, y_pred):
        y_true = y_true[..., :-1]
        y_pred = y_pred[..., :-1]
        return tf.losses.mean_squared_error(y_true, y_pred)

