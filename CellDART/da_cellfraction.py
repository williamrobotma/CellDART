import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.compat.v1 import disable_eager_execution, logging
from tensorflow.keras import optimizers
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.utils import to_categorical

# Suppress tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

logging.set_verbosity(logging.ERROR)
# Remove compatibility issues
disable_eager_execution()
experimental_run_tf_function = False


### Build deep learning models for adversarial domain adaptation
def build_models(inp_dim, emb_dim, n_cls_source, alpha=2, alpha_lr=10, bn_momentum=0.99):
    inputs = Input(shape=(inp_dim,))
    x4 = Dense(1024, activation="linear")(inputs)
    x4 = BatchNormalization(momentum=bn_momentum)(x4)
    x4 = Activation("elu")(x4)
    x4 = Dense(emb_dim, activation="linear")(x4)
    x4 = BatchNormalization(momentum=bn_momentum)(x4)
    x4 = Activation("elu")(x4)

    source_classifier = Dense(n_cls_source, activation="linear", name="mo1")(x4)
    source_classifier = Activation("softmax", name="mo")(source_classifier)

    domain_classifier = Dense(32, activation="linear", name="do4")(x4)
    domain_classifier = BatchNormalization(name="do5", momentum=bn_momentum)(domain_classifier)
    domain_classifier = Activation("elu", name="do6")(domain_classifier)
    domain_classifier = Dropout(0.5)(domain_classifier)
    domain_classifier = Dense(2, activation="softmax", name="do")(domain_classifier)

    comb_model = Model(inputs=inputs, outputs=[source_classifier, domain_classifier])
    comb_model.compile(
        optimizer="Adam",
        loss={"mo": "kld", "do": "categorical_crossentropy"},
        loss_weights={"mo": 1, "do": alpha},
        metrics=["accuracy"],
    )

    source_classification_model = Model(inputs=inputs, outputs=[source_classifier])
    source_classification_model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss={"mo": "kld"},
        metrics=["mae"],
    )

    domain_classification_model = Model(inputs=inputs, outputs=[domain_classifier])
    domain_classification_model.compile(
        optimizer=optimizers.Adam(learning_rate=alpha_lr * 0.001),
        loss={"do": "categorical_crossentropy"},
        metrics=["accuracy"],
    )

    embeddings_model = Model(inputs=inputs, outputs=[x4])
    embeddings_model.compile(
        optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return comb_model, source_classification_model, domain_classification_model, embeddings_model


### Batch data generator for the given data
def batch_generator(data, batch_size):
    """Generate batches of data.
    Given a list of numpy data, it iterates over the list and returns batches of the same size
    This
    """
    all_examples_indices = len(data[0])
    while True:
        mini_batch_indices = np.random.choice(all_examples_indices, size=batch_size, replace=False)
        tbr = [k[mini_batch_indices] for k in data]
        yield tbr


### Train the neural network to predict cell composition in spatial data
# Xs: numpy array for composite log-normaized count of pseudospots
# ys: numpy array for fraction of cell types across the pseudospots
# Xt: numpy array for log-normaized count of spatial spots

# emb_dim: output size of dimensions for feature extractor (default = 64)
# batch_size: minibatch size for pseudospots and spatial data during the training (default = 64)
# enable_dann: whether to use domain adaptation process
# n_iterations: iteration number for the adversarial learning (default = 3000)

# alpha: loss weights of domain classifier to the source classifier (default = 0.6)
# alpha_lr: learning rate for the domain classifier (alpha_lr*0.001, default = 5)


# init_train: whether to perform pre-training process
# init_train_epoch: iteration number for the pre-training process (default = 10)
def train(
    Xs,
    ys,
    Xt,
    yt=None,
    emb_dim=2,
    batch_size=64,
    batch_size_initial_train=None,
    enable_dann=True,
    n_iterations=1000,
    alpha=2,
    alpha_lr=10,
    initial_train=True,
    initial_train_epochs=100,
    bn_momentum=0.99,
    seed=None,
):
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    inp_dim = Xs.shape[1]
    ncls_source = ys.shape[1]

    (
        model,
        source_classification_model,
        domain_classification_model,
        embeddings_model,
    ) = build_models(
        inp_dim, emb_dim, ncls_source, alpha=alpha, alpha_lr=alpha_lr, bn_momentum=bn_momentum
    )

    if batch_size_initial_train is None:
        batch_size_initial_train = batch_size

    if initial_train:
        source_classification_model.fit(
            Xs, ys, batch_size=batch_size_initial_train, epochs=initial_train_epochs
        )
        print("initial_train_done")
        sourceloss, sourceacc = source_classification_model.evaluate(Xs, ys, verbose=0)
        print(sourceloss)

    source_classification_model_no_da = clone_model(source_classification_model)
    source_classification_model_no_da.set_weights(source_classification_model.get_weights())
    source_classification_model_no_da.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={"mo": "kld"},
        metrics=["mae"],
    )
    sourceloss, sourceacc = source_classification_model_no_da.evaluate(Xs, ys, verbose=0)
    print(sourceloss)
    embeddings_model_no_da = clone_model(embeddings_model)
    embeddings_model_no_da.set_weights(embeddings_model.get_weights())
    embeddings_model_no_da.compile(
        optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    y_adversarial_1 = to_categorical(np.array(([1] * batch_size + [0] * batch_size)))

    sample_weights_class = np.array(([1] * batch_size + [0] * batch_size))
    sample_weights_adversarial = np.ones((batch_size * 2,))

    S_batches = batch_generator([Xs, ys], batch_size)
    T_batches = batch_generator([Xt, np.zeros(shape=(len(Xt), 2))], batch_size)

    for i in range(n_iterations):
        # # print(y_class_dummy.shape, ys.shape)
        y_adversarial_2 = to_categorical(np.array(([0] * batch_size + [1] * batch_size)))

        X0, y0 = next(S_batches)
        X1, y1 = next(T_batches)

        X_adv = np.concatenate([X0, X1])
        y_class = np.concatenate([y0, np.zeros_like(y0)])

        adv_weights = []
        for layer in model.layers:
            if layer.name.startswith("do"):
                # print(layer.name)
                adv_weights.append(layer.get_weights())

        if enable_dann:
            # note - even though we save and append weights, the batchnorms moving means and variances
            # are not saved throught this mechanism
            model.train_on_batch(
                X_adv,
                [y_class, y_adversarial_1],
                sample_weight=[sample_weights_class, sample_weights_adversarial],
            )

            k = 0
            for layer in model.layers:
                if layer.name.startswith("do"):
                    layer.set_weights(adv_weights[k])
                    k += 1

            class_weights = []

            for layer in model.layers:
                if not layer.name.startswith("do"):
                    # if i == 0:
                    # print(layer.name)
                    class_weights.append(layer.get_weights())

            domain_classification_model.train_on_batch(X_adv, [y_adversarial_2])

            k = 0
            for layer in model.layers:
                if not layer.name.startswith("do"):
                    layer.set_weights(class_weights[k])
                    k += 1

        else:
            source_classification_model.train_on_batch(X0, y0)

        if yt is None:
            if (i + 1) % 100 == 0:
                # print(i, stats)
                sourceloss, sourceacc = source_classification_model.evaluate(Xs, ys, verbose=0)
                domainloss, domainacc = domain_classification_model.evaluate(
                    np.concatenate([Xs, Xt]),
                    to_categorical(np.array(([0] * Xs.shape[0] + [1] * Xt.shape[0]))),
                    verbose=0,
                )
                print(
                    "Iteration %d, source loss =  %.3f, discriminator acc = %.3f"
                    % (i, sourceloss, domainacc)
                )
        else:
            if (i + 1) % 100 == 0:
                # print(i, stats)
                y_test_hat_t = source_classification_model.predict(Xt).argmax(1)
                y_test_hat_s = source_classification_model.predict(Xs).argmax(1)
                print(
                    "Iteration %d, source accuracy =  %.3f, target accuracy = %.3f"
                    % (i, accuracy_score(ys, y_test_hat_s), accuracy_score(yt, y_test_hat_t))
                )

    return (
        embeddings_model,
        embeddings_model_no_da,
        source_classification_model,
        source_classification_model_no_da,
    )
