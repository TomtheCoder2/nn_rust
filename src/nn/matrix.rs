extern crate nalgebra as na;

use std::ops::{Sub};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

pub fn matrix_constructor(rows: usize, cols: usize) -> Matrix {
    Matrix {
        rows,
        cols,
        matrix: na::DMatrix::from_element(rows, cols, 0.0),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub matrix: na::DMatrix<f64>,
}


impl Default for Matrix {
    fn default() -> Matrix {
        Matrix {
            rows: 0,
            cols: 0,
            matrix: na::DMatrix::from_row_slice(0, 0, &[]),
        }
    }
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, seed: i32) -> Matrix {
        let mut m = Matrix {
            rows,
            cols,
            matrix: na::DMatrix::from_element(rows, cols, 0.0),
        };
        let mut r = StdRng::seed_from_u64(seed as u64);
        for i in 0..rows {
            for j in 0..cols {
                m.matrix[(i, j)] = r.gen_range(-1.0..1.0);
            }
        }
        m
    }

    pub fn new_from_matrix(m: na::DMatrix<f64>) -> Matrix {
        Matrix {
            rows: m.nrows(),
            cols: m.ncols(),
            matrix: m,
        }
    }

    pub fn check_equal(m1: &na::DMatrix<f64>, m2: &na::DMatrix<f64>) {
        for row in 0..m1.nrows() {
            for col in 0..m1.ncols() {
                assert_eq!(m1[(row, col)], m2[(row, col)]);
            }
        }
    }

    pub fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
        if a.cols != b.rows {
            panic!("Columns of A must match rows of B.");
        }
        let result = &a.matrix * &b.matrix;
        Matrix {
            rows: a.rows,
            cols: b.cols,
            matrix: result,
        }
    }

    // pub fn multiply_slow(a: &Matrix, b: &Matrix) -> Matrix {
    //     if a.cols != b.rows {
    //         panic!("Columns of A must match rows of B.");
    //     }
    //     let mut result = matrix_constructor(a.rows, b.cols);
    //     for i in 0..result.rows {
    //         for j in 0..result.cols {
    //             let mut sum = 0.0;
    //             for k in 0..a.cols {
    //                 sum += a.matrix[()] * b.data[k][j];
    //             }
    //             result.data[i][j] = sum;
    //         }
    //     }
    //     result
    // }


    pub fn dsigmoid(x: &Matrix) -> Matrix {
        let mut result = matrix_constructor(x.rows, x.cols);
        for i in 0..result.rows {
            for j in 0..result.cols {
                result.matrix[(i, j)] = x.matrix[(i, j)] * (1.0 - x.matrix[(i, j)]);
            }
        }
        result
    }

    pub fn scale(&mut self, scaler: f64) {
        self.matrix = &self.matrix * scaler
    }

    pub fn add_matrix(&mut self, m: &Matrix) {
        if self.rows != m.rows || self.cols != m.cols {
            panic!("Matrix add: matrices have different dimensions: {}x{} vs {}x{}",
                   self.rows,
                   self.cols,
                   m.rows,
                   m.cols);
        }
        // let mut old = self.matrix.clone();
        // for row in 0..self.rows {
        //     for col in 0..self.cols {
        //         self.matrix[(row, col)] += m.matrix[(row, col)];
        //     }
        // }
        // let tmp = self.matrix.clone();
        self.matrix = &self.matrix + &m.matrix;
        // Matrix::check_equal(&self.matrix, &old);
    }

    pub fn subtract(m1: &Matrix, m2: &Matrix) -> Matrix {
        if m1.rows != m2.rows || m1.cols != m2.cols {
            panic!("Matrix subtract: matrices have different dimensions");
        }
        Matrix::new_from_matrix(m1.matrix.clone().sub(&m2.matrix))
    }
    pub fn transpose(m1: &Matrix) -> Matrix {
        Matrix::new_from_matrix(m1.matrix.clone().transpose())
    }
    pub fn multiply_1to1(a: &Matrix, b: &Matrix) -> Matrix {
        let mut m = matrix_constructor(a.rows, a.cols);
        for rows in 0..a.rows {
            for cols in 0..a.cols {
                m.matrix[(rows, cols)] = a.matrix[(rows, cols)] * b.matrix[(rows, cols)];
            }
        }
        m
    }

    pub fn multiply_with_matrix(&mut self, m: &Matrix) {
        // if self.cols != m.rows {
        //     panic!("Matrix multiply: matrices have different dimensions (a: {}, b: {})", self.cols, self.rows);
        // }
        // for row in 0..self.rows {
        //     for col in 0..m.cols {
        //         self.matrix[(row, col)] *= m.matrix[(row, col)];
        //     }
        // }
        self.matrix = self.matrix.component_mul(&m.matrix);
    }
    pub fn multiply_with_double(&mut self, d: f64) {
        Matrix::new_from_matrix(self.matrix.clone().scale(d));
    }
    pub fn sigmoid(&mut self) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                self.matrix[(row, col)] = 1.0 / (1.0 + (-self.matrix[(row, col)]).exp());
            }
        }
    }

    pub fn sigmoid_derivative(&mut self) -> Matrix {
        let mut m = matrix_constructor(self.rows, self.cols);
        for row in 0..self.rows {
            for col in 0..self.cols {
                m.matrix[(row, col)] = self.matrix[(row, col)] * (1.0 - self.matrix[(row, col)]);
            }
        }
        m
    }
    pub fn from_array(arr: Vec<f64>) -> Matrix {
        let mut m = matrix_constructor(arr.len(), 1);
        for i in 0..arr.len() {
            m.matrix[(i, 0)] = arr[i];
        }
        m
    }
    pub fn from_2d_array(arr: Vec<Vec<f64>>) -> Matrix {
        let mut m: Matrix = matrix_constructor(arr.len(), arr[0].len());
        for i in 0..arr.len() {
            for j in 0..arr[i].len() {
                m.matrix[(i, j)] = arr[i][j];
            }
        }
        m
    }

    pub fn to_array(&self) -> Vec<f64> {
        let mut arr = vec![];
        for i in 0..self.rows {
            for j in 0..self.cols {
                arr.push(self.matrix[(i, j)]);
            }
        }
        arr
    }
}