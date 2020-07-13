import pytest
import numpy as np

from spatial_biofilm_sorting_package.binning import bin_a_from_b


def test_raise():
    with pytest.raises(ValueError):
        bin_a_from_b(
            np.zeros((2, 2)),
            np.zeros((3, 2)),
            10
        )


a = np.asarray([
    [0, 1, 2, 0, 1, 2, 0, 10, 20, 0, 10, 20],
    [0, 2, 4, 0, 2, 4, 0, 20, 40, 0, 20, 40],
]).transpose()
b = np.asarray([
    np.arange(12),
    np.linspace(0, 2, 12),
]).transpose()
ones = np.ones((2, 2))
zeros = np.zeros((2, 2))


def test_pdf():
    pdf = bin_a_from_b(a, b, 4)[2]
    correct_pdf = np.ones_like(pdf) * 0.25
    assert np.allclose(pdf, correct_pdf)


def test_mu():
    mu_sigma = bin_a_from_b(a, b, 4)[0]
    assert np.allclose(mu_sigma[0, :, :2, 0], ones)
    assert np.allclose(mu_sigma[0, :, 2:, 0], ones*10)
    assert np.allclose(mu_sigma[1, :, :2, 0], ones*2)
    assert np.allclose(mu_sigma[1, :, 2:, 0], ones*20)


def test_sigma():
    mu_sigma = bin_a_from_b(a, b, 4)[0]
    assert np.allclose(mu_sigma[0, :, :2, 1], ones)
    assert np.allclose(mu_sigma[0, :, 2:, 1], ones*10)
    assert np.allclose(mu_sigma[1, :, :2, 1], ones*2)
    assert np.allclose(mu_sigma[1, :, 2:, 1], ones*20)
