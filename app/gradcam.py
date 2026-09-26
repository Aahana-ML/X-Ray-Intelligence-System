import tensorflow as tf


def create_grad_model(model):

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer("top_conv").output,
            model.output
        ]
    )

    return grad_model



  

def make_gradcam(grad_model, image, class_index):

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(image)

        class_score = predictions[:, class_index]

    grads = tape.gradient(
        class_score,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(1, 2)
    )

    weighted_conv_outputs = (
        conv_outputs *
        pooled_grads[:, tf.newaxis, tf.newaxis, :]
    )

    heatmap = tf.reduce_sum(
        weighted_conv_outputs,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        (224, 224)
    )

    heatmap = tf.squeeze(
        heatmap
    )

    return heatmap.numpy()

    