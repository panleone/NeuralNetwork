#include <stdexcept>

#include<iostream>
#include<functional>
#include "matrix_tests.h"
#include "../matrix.h"

static void matrix_addition_tests();
static void matrix_product_tests();

template<typename Err>
void should_throw(const std::function<void()>& fn, const std::string& reason){
    bool success = false;
    try {
        fn();
        success = true;
    } catch(const Err& e){}
    if (success) throw std::runtime_error(reason);
}

void matrix_tests(){
    matrix_addition_tests();
    matrix_product_tests();
}

void check_matrix_equality(const Matrix<int>& m1, const Matrix<int>& m2, const std::string& reason){
    if(m1.N != m2.N || m1.M != m2.M) {
        throw std::runtime_error(reason);
    }
    for (size_t i  = 0; i < m1.N; i++){
        for (size_t j = 0; j < m1.M; j++){
            if (m1(i,j) != m2(i,j)){
                throw std::runtime_error(reason);
            }
        }
    }
}

void check_vector_equality(const Vector<int>& v1, const Vector<int>& v2, const std::string& reason){
    if(v1.N != v2.N || v1.M != v2.M || v1.M != 1) {
        throw std::runtime_error(reason);
    }
    for (size_t i  = 0; i < v1.N; i++){
        if (v1(i) != v2(i)){
                throw std::runtime_error(reason);
        }
    }
}

static void matrix_addition_tests(){
    Matrix<int> m1(2,2);
    m1(0,0) = 1;
    m1(1, 0)= 5;
    m1(0,1) = 6;
    m1(1,1) = 10;

    Matrix<int> m2(2,2);
    m2(0,0) = 7;
    m2(1, 0)= 9;
    m2(0,1) = -12;
    m2(1,1) = -24;

    Matrix<int> m_expected_sum(2,2);
    m_expected_sum(0,0) = 8;
    m_expected_sum(1,0) = 14;
    m_expected_sum(0,1) = -6;
    m_expected_sum(1,1) = -14;
    check_matrix_equality(m1 + m2, m_expected_sum, "Matrix sum is not working");
    m_expected_sum -= m2;
    check_matrix_equality(m1, m_expected_sum, "Matrix -= operator is not working");

    Matrix<int> m_expected_diff(2,2);
    m_expected_diff(0,0) = -6;
    m_expected_diff(1,0) = -4;
    m_expected_diff(0,1) = 18;
    m_expected_diff(1,1) = 34;
    check_matrix_equality(m1 - m2, m_expected_diff, "Matrix diff is not working");
    m_expected_diff += m2;
    check_matrix_equality(m1, m_expected_diff, "Matrix += operator is not working");

    // Test that summing different sizes throw
    Matrix<int> m_small(1,1);
    should_throw<MatrixException>([&](){m_small + m1;},"Summed matrices with different sizes");
    Matrix<int> m_big(3,3);
    should_throw<MatrixException>([&](){m_big - m1;},"Subtracted matrices with different sizes");

    // Repeat for vectors
    Vector<int> v1(2);
    v1(0) = 3;
    v1(1) = 2;
    Vector<int> v2(2);
    v2(0)= -5;
    v2(1) = -9;

    Vector<int> v_expected_sum(2);
    v_expected_sum(0) = -2;
    v_expected_sum(1) = -7;
    check_vector_equality(v1 + v2, v_expected_sum, "Vector sum is not working");
    v_expected_sum -= v2;
    check_vector_equality(v1, v_expected_sum, "Vector -= operator is not working");

    Vector<int> v_expected_diff(2);
    v_expected_diff(0) = 8;
    v_expected_diff(1) = 11;
    check_vector_equality(v1 - v2, v_expected_diff, "Vector diff is not working");
    v_expected_diff += v2;
    check_vector_equality(v1, v_expected_diff, "Vector += operator is not working");

    should_throw<MatrixException>([&](){m2 + v1;},"Summed matrices and vector");
    should_throw<MatrixException>([&](){m1 - v1;},"Subtracted matrices and vector");

    Vector<int> v3(3);
    should_throw<MatrixException>([&](){v3 + v1;},"Summed vectors of different sizes");
    should_throw<MatrixException>([&](){v3 - v1;},"Subtracted vectors of different sizes");
    should_throw<MatrixException>([&](){v3 += v1;},"+= operator applied on vectors of different sizes");
    should_throw<MatrixException>([&](){v3 -= v1;},"-= operator applied on vectors of different sizes");

    should_throw<MatrixException>([&](){v2 += m1;},"+= operator applied on vector and matrix");
    should_throw<MatrixException>([&](){v1 -= m1;},"-= operator applied on vector and matrix");
}

static void matrix_product_tests(){
    Matrix<int> m1(1,2);
    m1(0,0) = 1;
    m1(0, 1)= 5;

    Matrix<int> m2(2,2);
    m2(0,0) = 4;
    m2(1,0) = 7;
    m2(0, 1) = -1;
    m2(1,1) = -3;

    Matrix<int> m_expected_prod(1, 2);
    m_expected_prod(0, 0) = 39;
    m_expected_prod(0, 1) = -16;
    check_matrix_equality(m1*m2, m_expected_prod, "Matrix product doesn't work");
    should_throw<MatrixException>([&](){m2*m1;},"Product of matrices of incompatible sizes");

    Matrix<int> m_expected_prod_by_scalar(1,2);
    m_expected_prod_by_scalar(0,0) = 4;
    m_expected_prod_by_scalar(0, 1) = 20;

    check_matrix_equality(m1*4, m_expected_prod_by_scalar, "Matrix product by scalar doesn't work");
    check_matrix_equality(m1, m_expected_prod_by_scalar /= 4, "Matrix division by scalar /= operator doesn't work");
    check_matrix_equality(m1*4, m_expected_prod_by_scalar *= 4, "Matrix product by scalar *= operator doesn't work");

    Vector<int> v1(2);
    v1(0) = 5;
    v1(1) = 7;
    should_throw<MatrixException>([&](){v1*v1;},"Vectors has been multiplied");
    Matrix<int> mat_vec_prod = m2*v1;
    Vector<int> v_expected_mat_vec_prod(2);
    v_expected_mat_vec_prod(0) = 13;
    v_expected_mat_vec_prod(1) = 14;
    check_matrix_equality(mat_vec_prod, v_expected_mat_vec_prod, "Matrix vector product doesn't work");

    // Row matrix by vector multiplication
    Matrix<int> scalar = m1*v1;
    Matrix<int> scalar_expected(1,1);
    scalar_expected(0,0) = 40;
    check_matrix_equality(scalar, scalar_expected, "Matrix vector product doesn't work");

    // Check matrix-vector conversion
    Vector<int> vector_cast = std::move(mat_vec_prod);
    check_vector_equality(vector_cast, v_expected_mat_vec_prod, "Casting matrix to vector doesn't work");
    should_throw<MatrixException>([&](){vector_cast = std::move(m1);},"Casted diagonal matrix to vector");

    // Check clone operator
    Matrix<int> cached_m2(2,2);
    cached_m2(0,0) = 4;
    cached_m2(1,0) = 7;
    cached_m2(0, 1) = -1;
    cached_m2(1,1) = -3;

    Matrix<int> matrix_clone = m2.clone();
    check_matrix_equality(matrix_clone, cached_m2, "Matrix clone doesn't work");
    check_matrix_equality(cached_m2, m2, "Matrix clone doesn't work");
    check_matrix_equality(matrix_clone, m2, "Matrix clone doesn't work");


    Vector<int> v1_cache(2);
    v1_cache(0) = 5;
    v1_cache(1) = 7;
    Vector<int> v1_clone = v1.clone();
    check_vector_equality(v1_cache, v1, "Vector clone doesn't work");
    check_vector_equality(v1_clone, v1_cache, "Vector clone doesn't work");
    check_vector_equality(v1_clone, v1, "Vector clone doesn't work");
}