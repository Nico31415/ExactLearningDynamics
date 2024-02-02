from ..environment import env

import jax
import jax.numpy as np
import numpy as onp

from functools import partial


class Dataset:
    def __new__(cls, *args, **kwargs):
        cls.training_data = np.array([])
        cls.training_labels = np.array([])
        cls.validation_data = np.array([])
        cls.validation_labels = np.array([])
        cls.test_data = np.array([])
        cls.test_labels = np.array([])

        return cls

    def create_advances(cls, preprocessing):
        x_train, x_validate, x_test = cls.training_data, cls.validation_data, cls.test_data
        y_train, y_validate, y_test = cls.training_labels, cls.validation_labels, cls.test_labels
        for fnct in preprocessing:
            x_train, x_validate, x_test, y_train, y_validate, y_test = fnct(x_train, x_validate, x_test, y_train,
                                                                            y_validate, y_test)


        train = partial(cls.advance_training, data=x_train, labels=y_train)
        validate = partial(cls.advance_training, data=x_validate, labels=y_validate)
        test = partial(cls.advance_training, data=x_test, labels=y_test)
        return train, validate, test

    @staticmethod
    def advance_training(idx, data, labels):
        if idx is None:
            return data, labels
        return data[idx], labels[idx]

    @staticmethod
    def advance_validation(idx, data, labels):
        pass

    @staticmethod
    def advance_test(idx, data, labels):
        pass



## TODO use this
class RandomRegression(Dataset):
    def __new__(cls, items_n, input_dim, output_dim, preprocessing=[]):
        cls = super().__new__(cls)
        keys = env.create_rng_keys(2)

        cls.training_data = jax.random.normal(keys[0], (items_n, input_dim)) * 1. / np.sqrt(input_dim)
        cls.training_labels = jax.random.normal(keys[1], (items_n, output_dim)) * 1. / np.sqrt(output_dim)

        return cls.create_advances(cls, preprocessing)


class ReversalLearning(Dataset):
    def __new__(cls, items_n, input_dim, output_dim, preprocessing=[]):
        cls = super().__new__(cls)
        keys = env.create_rng_keys(2)

        R, _, _ = np.linalg.svd(jax.random.normal(keys[0], (items_n, items_n)))
        U, S, VT = np.linalg.svd(jax.random.normal(keys[1], (output_dim, input_dim)))
        SS = np.diag(S * np.sqrt(1. / S)) # *  1. / np.sqrt(np.max(np.asarray([input_dim, output_dim])))

        smaller_dim = input_dim if input_dim < output_dim else output_dim

        S0 = np.vstack([SS, np.zeros((items_n - smaller_dim, smaller_dim))])
        cls.training_data = R @ S0 @ VT
        S0 = np.hstack([SS, np.zeros((smaller_dim, items_n - smaller_dim))])
        cls.training_labels = (U @ S0 @ R.T).T

        cls.validation_data = cls.training_data.copy()
        cls.validation_labels = (-U @ S0 @ R.T).T

        return cls.create_advances(cls, preprocessing)


class StudentTeacher(Dataset):
    def __new__(cls, items_n, ws, preprocessing=[]):
        cls = super().__new__(cls)
        key = env.create_rng_keys(1)[0]
        input_dim = ws[0].shape[-1]
        cls.training_data = jax.random.normal(key, (items_n, input_dim))
        x_copy = cls.training_data.copy()

        train, _, _ = cls.create_advances(cls, preprocessing)
        x = train(None)[0].T

        for w in ws:
            x = np.asarray(w) @ x

        cls.training_data = x_copy
        cls.training_labels = x.T

        return cls.create_advances(cls, preprocessing)

class Hierarchy(Dataset):
    def __new__(cls, reversal, correct, preprocessing=[]):
        cls = super().__new__(cls)

        cls.training_data = np.identity(8)

        cls.training_labels = np.asarray([
            [1.,  1.,  1., -0.,  1., -0., -0., -0.],
            [1.,  1.,  1., -0., -1., -0., -0., -0.],
            [1.,  1., -1., -0., -0.,  1., -0., -0.],
            [1.,  1., -1., -0., -0., -1., -0., -0.],
            [1., -1., -0.,  1., -0., -0.,  1., -0.],
            [1., -1., -0.,  1., -0., -0., -1., -0.],
            [1., -1., -0., -1., -0., -0., -0.,  1.],
            [1., -1., -0., -1., -0., -0., -0., -1.]
        ])


        if correct:
             cls.training_labels *= 1. + np.asarray([[0.1 / i for i in range(1, 9)]])

        if reversal:
            cls.training_labels = cls.training_labels * np.concatenate([np.ones(7), -1*np.ones(1)])

        return cls.create_advances(cls, preprocessing)


class ColourHierarchy(Dataset):
    def __new__(cls, correct, preprocessing=[]):
        cls = super().__new__(cls)

        cls.training_data = np.identity(8)

        cls.training_labels = np.asarray([
            [1., -1.,  0.,  1.,  0.,  0.,  1.,  0.],
            [1.,  1., -1.,  0.,  0., -1.,  0.,  0.],
            [1.,  1.,  1.,  0.,  1.,  0.,  0.,  0.],
            [1., -1.,  0., -1.,  0.,  0.,  0.,  1.],
            [1.,  1., -1.,  0.,  0.,  1.,  0.,  0.],
            [1.,  1.,  1.,  0., -1.,  0.,  0.,  0.],
            [1., -1.,  0., -1.,  0.,  0.,  0., -1.],
            [1., -1.,  0.,  1.,  0.,  0., -1.,  0.]
        ])

        if correct:
             cls.training_labels *= 1. + np.asarray([[0.1 / i for i in range(1, 9)]])

        return cls.create_advances(cls, preprocessing)


class Whiten:
    def __new__(cls, threshold=1e-12):
        return partial(cls.advance, threshold=threshold)

    @staticmethod
    def advance(x_train, x_validate, x_test, y_train, y_validate, y_test, threshold):
        """
        if np.all(x_train == np.identity(x_train.shape[0])):
            x_train /= x_train.shape[0]
            if len(x_validate) > 0:
                x_validate /= x_train.shape[0]
            if len(x_test) > 0:
                x_test /= x_train.shape[0]
            return x_train, x_validate, x_test, y_train, y_validate, y_test
        """
        x_train = x_train.T - np.mean(x_train.T, axis=1, keepdims=True)

        eig_vals, eig_vecs = onp.linalg.eig(np.cov(x_train, bias=True))

        idx = onp.abs(eig_vals) < threshold
        eig_vals[idx] = 1.
        eig_vecs[idx] = 0.
        eig_vals = np.real(eig_vals)
        eig_vecs = np.real(eig_vecs)

        x_train = (np.diag(1. / np.sqrt(eig_vals)) @ eig_vecs.T @ x_train).T
        if len(x_validate) > 0:
            x_validate = x_validate.T - np.mean(x_validate.T, axis=1, keepdims=True)
            x_validate = (np.diag(1. / np.sqrt(eig_vals)) @ eig_vecs.T @ x_validate).T
        if len(x_test) > 0:
            x_test = x_test.T - np.mean(x_test.T, axis=1, keepdims=True)
            x_test = (np.diag(1. / np.sqrt(eig_vals)) @ eig_vecs.T @ x_test).T

        return x_train, x_validate, x_test, y_train, y_validate, y_test

