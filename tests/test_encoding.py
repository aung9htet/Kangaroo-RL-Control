import numpy as np
import torch

from kangaroo_rl.encoding import encoded_size, normalise, positional_encoding, positional_encoding_torch


def test_shape_and_layout(rng):
    x = rng.uniform(-1, 1, (5, 7))
    features = positional_encoding(x, 4)
    assert features.shape == (5, encoded_size(7, 4)) == (5, 63)
    np.testing.assert_allclose(features[:, :7], x)                       # raw value first
    np.testing.assert_allclose(features[:, 7:14], np.sin(np.pi * x))     # then sin(2^0 pi x)
    np.testing.assert_allclose(features[:, 14:21], np.cos(np.pi * x))
    np.testing.assert_allclose(features[:, -7:], np.cos(8 * np.pi * x))  # last: cos(2^3 pi x)


def test_raw_is_identity(rng):
    x = rng.uniform(-1, 1, (3, 7))
    np.testing.assert_allclose(positional_encoding(x, 0), x)


def test_numpy_and_torch_agree(rng):
    x = rng.uniform(-1, 1, (4, 3))
    np.testing.assert_allclose(positional_encoding_torch(torch.tensor(x), 4).numpy(), positional_encoding(x, 4),
                               atol=1e-12)


def test_normalise_maps_limits_to_unit_interval():
    lower, upper = np.array([-2.0, 0.0]), np.array([2.0, 1.0])
    np.testing.assert_allclose(normalise(lower, lower, upper), [-1, -1])
    np.testing.assert_allclose(normalise(upper, lower, upper), [1, 1])


def test_encoding_separates_nearby_angles():
    """Higher frequencies make a 0.005 rad step (one fine bin) visible in feature space."""
    scale = 2 / (2 * 2.618)                 # normalised units per radian for a +-2.618 rad joint
    a, b = np.array([[0.1]]), np.array([[0.1 + 0.005 * scale]])
    raw = np.linalg.norm(positional_encoding(b, 0) - positional_encoding(a, 0))
    rich = np.linalg.norm(positional_encoding(b, 4) - positional_encoding(a, 4))
    assert rich > 5 * raw
