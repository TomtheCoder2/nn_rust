extern crate nalgebra as na;

use nn_rust::nn::matrix::Matrix;

fn main() {
    let n = 20;
    let m = 30;
    let k = 40;
    let a = Matrix::new(n, m, 12);
    let b = Matrix::new(m, k, 12);
    let mut c = Matrix::new(n, k, 12);
    // start timer
    let start = std::time::Instant::now();
    for _ in 0..1_000_000 {
        let c = Matrix::multiply(&a, &b);
    }
    // stop timer
    let duration = start.elapsed();
    println!("Time elapsed in multiply() is: {:?}", duration);
    // create na:Matrices
    // start timer
    let start = std::time::Instant::now();
    for _ in 0..1_000_000 {
        let result_m = &a.matrix * &b.matrix;
    }
    // stop timer
    let duration = start.elapsed();
    println!("Time elapsed in nalgebra is: {:?}", duration);

    // benchmark addition
    // start timer
    let b = Matrix::new(n, m, 12);
    let start = std::time::Instant::now();
    for _ in 0..1_000_000 {
        let result_m = &a.matrix + &b.matrix;
    }
    // stop timer
    let duration = start.elapsed();
    println!("Time elapsed in + is: {:?}", duration);

    // start timer
    let start = std::time::Instant::now();
    for _ in 0..1_000_000 {
        let result_m = a.clone().add_matrix(&b);
    }
    // stop timer
    let duration = start.elapsed();
    println!("Time elapsed in add_matrix is: {:?}", duration);
}