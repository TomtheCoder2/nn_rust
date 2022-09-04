use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

pub fn matrix_constructor(rows: usize, cols: usize) -> Matrix {
    Matrix {
        rows,
        cols,
        data: vec![vec![0.0; cols]; rows],
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

pub fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
    if a.cols != b.rows {
        panic!("Columns of A must match rows of B.");
    }
    let mut result = matrix_constructor(a.rows, b.cols);
    for i in 0..result.rows {
        for j in 0..result.cols {
            let mut sum = 0.0;
            for k in 0..a.cols {
                sum += a.data[i][k] * b.data[k][j];
            }
            result.data[i][j] = sum;
        }
    }
    result
}

pub fn dsigmoid(x: &Matrix) -> Matrix {
    let mut result = matrix_constructor(x.rows, x.cols);
    for i in 0..result.rows {
        for j in 0..result.cols {
            result.data[i][j] = x.data[i][j] * (1.0 - x.data[i][j]);
        }
    }
    result
}

impl Default for Matrix {
    fn default() -> Matrix {
        Matrix {
            rows: 0,
            cols: 0,
            data: vec![],
        }
    }
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, seed: i32) -> Matrix {
        let mut m = Matrix {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        };
        let mut r = StdRng::seed_from_u64(seed as u64);
        for i in 0..rows {
            for j in 0..cols {
                m.data[i][j] = r.gen_range(-1.0..1.0);
            }
        }
        m
    }

    pub fn scale(&mut self, scaler: f64) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                self.data[row][col] += scaler;
            }
        }
    }
    pub fn add_matrix(&mut self, m: &Matrix) {
        if self.rows != m.rows || self.cols != m.cols {
            panic!("Matrix add: matrices have different dimensions: {}x{} vs {}x{}",
                   self.rows,
                   self.cols,
                   m.rows,
                   m.cols);
        }
        for row in 0..self.rows {
            for col in 0..self.cols {
                self.data[row][col] += m.data[row][col];
            }
        }
    }
    pub fn subtract(m1: &Matrix, m2: &Matrix) -> Matrix {
        if m1.rows != m2.rows || m1.cols != m2.cols {
            panic!("Matrix subtract: matrices have different dimensions");
        }
        let mut m = matrix_constructor(m1.rows, m1.cols);
        m.data = vec![vec![0.0; m1.cols]; m1.rows];
        for row in 0..m1.rows {
            for col in 0..m1.cols {
                m.data[row][col] = m1.data[row][col] - m2.data[row][col];
            }
        }
        m
    }
    pub fn transpose(m1: &Matrix) -> Matrix {
        let mut m: Matrix = matrix_constructor(m1.cols, m1.rows);
        m.data = vec![vec![0.0; m1.cols]; m1.rows];
        for row in 0..m1.rows {
            for col in 0..m1.cols {
                m.data[col][row] = m1.data[row][col];
            }
        }
        m
    }
    pub fn multiply_two(m1: Matrix, m2: Matrix) -> Matrix {
        if m1.cols != m2.rows {
            panic!("Matrix multiply: matrices have different dimensions");
        }
        let mut m = matrix_constructor(m1.rows, m2.cols);
        m.data = vec![vec![0.0; m2.cols]; m1.rows];
        for row in 0..m1.rows {
            for col in 0..m2.cols {
                let mut sum: f64 = 0.0;
                for i in 0..m1.cols {
                    sum += m1.data[row][i] * m2.data[i][col];
                }
                m.data[row][col] = sum;
            }
        }
        m
    }
    pub fn multiply_with_matrix(&mut self, m: &Matrix) {
        if self.cols != m.rows {
            panic!("Matrix multiply: matrices have different dimensions");
        }
        for row in 0..self.rows {
            for col in 0..m.cols {
                self.data[row][col] *= m.data[row][col];
            }
        }
    }
    pub fn multiply_with_double(&mut self, d: f64) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                self.data[row][col] *= d;
            }
        }
    }
    pub fn sigmoid(&mut self) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                self.data[row][col] = 1.0 / (1.0 + (-self.data[row][col]).exp());
            }
        }
    }
    pub fn dsigmoid(&mut self) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                self.data[row][col] = self.data[row][col] * (1.0 - self.data[row][col]);
            }
        }
    }

    pub fn sigmoid_derivative(&mut self) -> Matrix {
        let mut m = matrix_constructor(self.rows, self.cols);
        m.data = vec![vec![0.0; self.cols]; self.rows];
        for row in 0..self.rows {
            for col in 0..self.cols {
                m.data[row][col] = self.data[row][col] * (1.0 - self.data[row][col]);
            }
        }
        m
    }
    pub fn from_array(arr: Vec<f64>) -> Matrix {
        let mut m = matrix_constructor(arr.len(), 1);
        for i in 0..arr.len() {
            m.data[i][0] = arr[i];
        }
        m
    }
    pub fn from_2d_array(arr: Vec<Vec<f64>>) -> Matrix {
        let mut m: Matrix = matrix_constructor(arr.len(), arr[0].len());
        for i in 0..arr.len() {
            for j in 0..arr[i].len() {
                m.data[i][j] = arr[i][j];
            }
        }
        m
    }

    pub fn to_array(&self) -> Vec<f64> {
        let mut arr = vec![];
        for i in 0..self.rows {
            for j in 0..self.cols {
                arr.push(self.data[i][j]);
            }
        }
        arr
    }
}