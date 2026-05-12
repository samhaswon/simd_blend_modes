import math

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial


def horner(coeffs, x):
    """Evaluate a polynomial using Horner's method.

    :param coeffs: Coefficients in ascending order.
    :param x: Input value.
    :return: Polynomial value.
    """
    result = 0.0

    for coeff in reversed(coeffs):
        result = result * x + coeff

    return result


def fit_forward_pow_2_4(degree=7, samples=20_001):
    """Fit x**2.4 on [0, 1].

    :param degree: Polynomial degree.
    :param samples: Number of samples used for fitting.
    :return: Polynomial coefficients in ascending order.
    """
    x_samples = np.linspace(0.0, 1.0, samples)
    y_samples = x_samples**2.4

    cheb_fit = Chebyshev.fit(
        x_samples,
        y_samples,
        deg=degree,
        domain=[0.0, 1.0],
    )

    return cheb_fit.convert(kind=Polynomial).coef


def fit_inverse_mantissa_pow_5_12(degree=5, samples=20_001):
    """Fit m**(5/12) on m in [0.5, 1].

    :param degree: Polynomial degree.
    :param samples: Number of samples used for fitting.
    :return: Polynomial coefficients in ascending order.
    """
    m_samples = np.linspace(0.5, 1.0, samples)
    y_samples = m_samples ** (5.0 / 12.0)

    cheb_fit = Chebyshev.fit(
        m_samples,
        y_samples,
        deg=degree,
        domain=[0.5, 1.0],
    )

    return cheb_fit.convert(kind=Polynomial).coef


def approx_forward_pow_2_4(x, coeffs):
    """Approximate x**2.4 on [0, 1].

    :param x: Input value.
    :param coeffs: Polynomial coefficients for x**2.4.
    :return: Approximate x**2.4.
    """
    if x <= 0.0:
        return 0.0

    if x >= 1.0:
        return 1.0

    return horner(coeffs, x)


def approx_inverse_pow_5_12(x, coeffs, scale_table):
    """Approximate x**(5/12) on [0, 1].

    :param x: Input value.
    :param coeffs: Polynomial coefficients for m**(5/12), m in [0.5, 1].
    :param scale_table: Table where table[r] = 2**(5*r/12).
    :return: Approximate x**(5/12).
    """
    if x <= 0.0:
        return 0.0

    if x >= 1.0:
        return 1.0

    mantissa, exponent = math.frexp(x)

    quotient, remainder = divmod(exponent, 12)

    mantissa_part = horner(coeffs, mantissa)
    scale_part = scale_table[remainder]

    return math.ldexp(mantissa_part * scale_part, 5 * quotient)


def report_error(func, actual_func, checks=1_000_001):
    """Report max absolute error on [0, 1].

    :param func: Approximation function.
    :param actual_func: Exact function.
    :param checks: Number of test points.
    :return: Tuple of max absolute error, x location, and signed error.
    """
    x_check = np.linspace(0.0, 1.0, checks)
    approx = np.array([func(float(x)) for x in x_check])
    actual = actual_func(x_check)

    error = approx - actual
    max_index = np.argmax(np.abs(error))

    return abs(error[max_index]), x_check[max_index], error[max_index]


def print_coefficients(name, coeffs):
    """Print polynomial coefficients.

    :param name: Polynomial name.
    :param coeffs: Coefficients in ascending order.
    """
    print(name)
    print("For c0 + c1*x + c2*x^2 + ...:")

    for index, coeff in enumerate(coeffs):
        print(f"c{index} = {coeff:.17g}")

    print()


def main():
    forward_degree = 7
    inverse_degrees = [3, 4, 5, 6]

    forward_coeffs = fit_forward_pow_2_4(degree=forward_degree)
    scale_table = np.array([2.0 ** (5.0 * r / 12.0) for r in range(12)])

    print_coefficients(
        f"x^2.4 forward polynomial, degree {forward_degree}",
        forward_coeffs,
    )

    forward_error, forward_x, forward_signed = report_error(
        lambda x: approx_forward_pow_2_4(x, forward_coeffs),
        lambda x: x**2.4,
    )

    print("x^2.4 forward error")
    print(f"max_abs_error = {forward_error:.17g}")
    print(f"x_at_max_error = {forward_x:.17g}")
    print(f"signed_error = {forward_signed:.17g}")
    print()

    print("inverse scale table")
    print("table[r] = 2^(5*r/12)")
    for index, value in enumerate(scale_table):
        print(f"table[{index}] = {value:.17g}")
    print()

    for inverse_degree in inverse_degrees:
        inverse_coeffs = fit_inverse_mantissa_pow_5_12(degree=inverse_degree)

        print_coefficients(
            f"x^(1/2.4) inverse mantissa polynomial, degree {inverse_degree}",
            inverse_coeffs,
        )

        inverse_error, inverse_x, inverse_signed = report_error(
            lambda x: approx_inverse_pow_5_12(x, inverse_coeffs, scale_table),
            lambda x: x ** (1.0 / 2.4),
        )

        print(f"x^(1/2.4) inverse error, degree {inverse_degree}")
        print(f"max_abs_error = {inverse_error:.17g}")
        print(f"x_at_max_error = {inverse_x:.17g}")
        print(f"signed_error = {inverse_signed:.17g}")
        print()


if __name__ == "__main__":
    main()
