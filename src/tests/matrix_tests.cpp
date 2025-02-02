#include <functional>
#include <stdexcept>

#include "../matrix.h"
#include "matrix_tests.h"
#include "test_utils.h"

static void matrix_addition_tests();
static void matrix_product_tests();
static void transpose_tests();
static void outer_product_tests();
static void initializer_list_tests();

void matrix_tests() {
  initializer_list_tests();
  matrix_addition_tests();
  matrix_product_tests();
  transpose_tests();
  outer_product_tests();
}

static void check_matrix_equality(const Matrix<int> &m1, const Matrix<int> &m2,
                                  const std::string &reason) {
  check_matrix_equality<int>(m1, m2, 0, reason);
}
static void check_vector_equality(const Vector<int> &v1, const Vector<int> &v2,
                                  const std::string &reason) {
  check_vector_equality<int>(v1, v2, 0, reason);
}

static void matrix_addition_tests() {
  Matrix<int> m1{{1, 6}, {5, 10}};

  Matrix<int> m2{{7, -12}, {9, -24}};

  Matrix<int> m_expected_sum{{8, -6}, {14, -14}};
  check_matrix_equality(m1 + m2, m_expected_sum, "Matrix sum is not working");
  m_expected_sum -= m2;
  check_matrix_equality(m1, m_expected_sum,
                        "Matrix -= operator is not working");

  Matrix<int> m_expected_diff{{-6, 18}, {-4, 34}};
  check_matrix_equality(m1 - m2, m_expected_diff, "Matrix diff is not working");
  m_expected_diff += m2;
  check_matrix_equality(m1, m_expected_diff,
                        "Matrix += operator is not working");

  // Test that summing different sizes throw
  Matrix<int> m_small(1, 1);
  should_throw<MatrixException>([&]() { m_small + m1; },
                                "Summed matrices with different sizes");
  Matrix<int> m_big(3, 3);
  should_throw<MatrixException>([&]() { m_big - m1; },
                                "Subtracted matrices with different sizes");

  // Repeat for vectors
  Vector<int> v1{3, 2};
  Vector<int> v2{-5, -9};

  Vector<int> v_expected_sum{-2, -7};
  check_vector_equality(v1 + v2, v_expected_sum, "Vector sum is not working");
  v_expected_sum -= v2;
  check_vector_equality(v1, v_expected_sum,
                        "Vector -= operator is not working");

  Vector<int> v_expected_diff{8, 11};
  check_vector_equality(v1 - v2, v_expected_diff, "Vector diff is not working");
  v_expected_diff += v2;
  check_vector_equality(v1, v_expected_diff,
                        "Vector += operator is not working");

  should_throw<MatrixException>([&]() { m2 + v1; },
                                "Summed matrices and vector");
  should_throw<MatrixException>([&]() { m1 - v1; },
                                "Subtracted matrices and vector");

  Vector<int> v3(3);
  should_throw<MatrixException>([&]() { v3 + v1; },
                                "Summed vectors of different sizes");
  should_throw<MatrixException>([&]() { v3 - v1; },
                                "Subtracted vectors of different sizes");
  should_throw<MatrixException>(
      [&]() { v3 += v1; }, "+= operator applied on vectors of different sizes");
  should_throw<MatrixException>(
      [&]() { v3 -= v1; }, "-= operator applied on vectors of different sizes");

  should_throw<MatrixException>([&]() { v2 += m1; },
                                "+= operator applied on vector and matrix");
  should_throw<MatrixException>([&]() { v1 -= m1; },
                                "-= operator applied on vector and matrix");
}

static void matrix_product_tests() {
  Matrix<int> m1{{1, 5}};

  Matrix<int> m2{{4, -1}, {7, -3}};

  Matrix<int> m_expected_prod{{39, -16}};
  check_matrix_equality(m1 * m2, m_expected_prod,
                        "Matrix product doesn't work");
  should_throw<MatrixException>([&]() { m2 *m1; },
                                "Product of matrices of incompatible sizes");

  Matrix<int> m_expected_prod_by_scalar{{4, 20}};

  check_matrix_equality(m1 * 4, m_expected_prod_by_scalar,
                        "Matrix product by scalar doesn't work");
  check_matrix_equality(m1, m_expected_prod_by_scalar /= 4,
                        "Matrix division by scalar /= operator doesn't work");
  check_matrix_equality(m1 * 4, m_expected_prod_by_scalar *= 4,
                        "Matrix product by scalar *= operator doesn't work");

  Vector<int> v1{5, 7};
  should_throw<MatrixException>([&]() { v1 *v1; },
                                "Vectors has been multiplied");
  Matrix<int> mat_vec_prod = m2 * v1;
  Vector<int> v_expected_mat_vec_prod{13, 14};
  check_matrix_equality(mat_vec_prod, v_expected_mat_vec_prod,
                        "Matrix vector product doesn't work");

  // Row matrix by vector multiplication
  Matrix<int> scalar = m1 * v1;
  Matrix<int> scalar_expected{{40}};
  check_matrix_equality(scalar, scalar_expected,
                        "Matrix vector product doesn't work");

  // Check matrix-vector conversion
  Vector<int> vector_cast = std::move(mat_vec_prod);
  check_vector_equality(vector_cast, v_expected_mat_vec_prod,
                        "Casting matrix to vector doesn't work");
  should_throw<MatrixException>([&]() { vector_cast = std::move(m1); },
                                "Casted diagonal matrix to vector");

  // Check clone operator
  Matrix<int> cached_m2{{4, -1}, {7, -3}};

  Matrix<int> matrix_clone = m2.clone();
  check_matrix_equality(matrix_clone, cached_m2, "Matrix clone doesn't work");
  check_matrix_equality(cached_m2, m2, "Matrix clone doesn't work");
  check_matrix_equality(matrix_clone, m2, "Matrix clone doesn't work");

  Vector<int> v1_cache{5, 7};
  Vector<int> v1_clone = v1.clone();
  check_vector_equality(v1_cache, v1, "Vector clone doesn't work");
  check_vector_equality(v1_clone, v1_cache, "Vector clone doesn't work");
  check_vector_equality(v1_clone, v1, "Vector clone doesn't work");
}

static void transpose_tests() {
  Matrix<int> m1{{1, 2, 3}, {4, 5, 6}};

  Matrix<int> m1_transpose{{1, 4}, {2, 5}, {3, 6}};
  check_matrix_equality(m1.transpose(), m1_transpose,
                        "Matrix transpose doesn't work");

  Vector<int> v1(2);
  v1(0) = 5;
  v1(1) = 7;

  Matrix<int> v1_transpose(1, 2);
  v1_transpose(0, 0) = 5;
  v1_transpose(0, 1) = 7;
  check_matrix_equality(v1.transpose(), v1_transpose,
                        "Vector transpose doesn't work");
}

static void outer_product_tests() {
  Vector<int> v1{5, 7};
  Vector<int> v2{1, 0, -5};

  Matrix<int> m{{5, 0, -25}, {7, 0, -35}};

  check_matrix_equality(outerProduct(v1, v2), m,
                        "Vector outer product doesn't work");
}

static void initializer_list_tests() {
  Matrix<int> mat{{1, 2}, {3, 4}};

  Matrix<int> expected(2, 2);
  expected(0, 0) = 1;
  expected(0, 1) = 2;
  expected(1, 0) = 3;
  expected(1, 1) = 4;

  check_matrix_equality(mat, expected, "Matrix constructor is broken");
  should_throw<MatrixException>(
      [] {
        Matrix<int>{{1}, {1, 2}};
      },
      "Can create a matrix with different number of rows");
  should_throw<MatrixException>(
      [] {
        Matrix<int>{{}, {}};
      },
      "Can create a matrix with empty rows");

  Vector<int> v{1, 0, -5};
  Vector<int> expectedV(3);
  expectedV(0) = 1;
  expectedV(1) = 0;
  expectedV(2) = -5;

  check_vector_equality(v, expectedV, "Vector constructor is broken");
}