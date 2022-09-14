use rand::seq::SliceRandom;
use crate::nn::matrix::Matrix;

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    pub layer_sizes: Vec<usize>,
    pub weights: Vec<Matrix>,
    pub biases: Vec<Matrix>,
    pub learning_rate: f64,
}

impl NeuralNetwork {
    pub fn new(layer_sizes: Vec<usize>, learning_rate: f64, seed: i32) -> NeuralNetwork {
        let mut weights: Vec<Matrix> = Vec::new();
        let mut biases: Vec<Matrix> = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            let weight = Matrix::new(layer_sizes[i + 1], layer_sizes[i], seed);
            let bias = Matrix::new(layer_sizes[i + 1], 1, seed);
            weights.push(weight);
            biases.push(bias);
        }
        NeuralNetwork {
            layer_sizes,
            weights,
            biases,
            learning_rate,
        }
    }

    pub fn predict(&self, input: Vec<f64>) -> Vec<f64> {
        let mut layers: Vec<Matrix> = Vec::new();
        layers.push(Matrix::from_array(input));

        for i in 0..self.layer_sizes.len() - 1 {
            let mut layer: Matrix = Matrix::multiply(&self.weights[i], &layers[i]);
            layer.add_matrix(&self.biases[i]);
            layer.sigmoid();
            layers.push(layer);
        }
        layers[layers.len() - 1].to_array()
    }

    pub fn train(&mut self, input: Vec<f64>, target_v: Vec<f64>) -> Vec<f64> {
        let mut layers: Vec<Matrix> = Vec::new();
        layers.push(Matrix::from_array(input));

        for i in 0..self.layer_sizes.len() - 1 {
            let mut layer: Matrix = Matrix::multiply(&self.weights[i], &layers[i]);
            layer.add_matrix(&self.biases[i]);
            layer.sigmoid();
            layers.push(layer);
        }

        let target = Matrix::from_array(target_v);
        let mut error = Matrix::subtract(&target, &layers[layers.len() - 1]);
        let mut transposed: Matrix;
        self.correct_error(layers.len() - 1, &layers, &error);
        for i in (1..self.layer_sizes.len() - 1).rev() {
            transposed = Matrix::transpose(&self.weights[i]);
            error = Matrix::multiply(&transposed, &error);
            self.correct_error(i, &layers, &error);
        }
        layers[layers.len() - 1].to_array()
    }

    fn correct_error(&mut self, i: usize, layers: &Vec<Matrix>, error: &Matrix) {
        let mut h_gradient: Matrix = Matrix::dsigmoid(&layers[i]);
        h_gradient.multiply_with_matrix(error);
        h_gradient.multiply_with_double(self.learning_rate);

        let wih_delta = Matrix::multiply(&h_gradient, &Matrix::transpose(&layers[i - 1]));
        self.weights[i - 1].add_matrix(&wih_delta);
        self.biases[i - 1].add_matrix(&h_gradient);
    }

    pub fn fit(&mut self, inputs: &Vec<Vec<f64>>, targets: &Vec<Vec<f64>>, epochs: i32) {
        for _e in 0..epochs {
            let mut samples: Vec<usize> = (0..inputs.len()).collect();
            samples.shuffle(&mut rand::thread_rng());

            let mut _errors_this_epoch = 0;
            for i in samples {
                // println!("input: {:?}", inputs[i].clone());
                // println!("target: {:?}", targets[i].clone());
                let _result = self.train(inputs[i].clone(), targets[i].clone());
                // check result
                // let bit_result: Vec<i32> = result.iter().map(|x| if *x > 0.5 { 1 } else { 0 }).collect();
                // for j in 0..bit_result.len() {
                //     if bit_result[j] != targets[i][j] as i32 {
                //         _errors_this_epoch += 1;
                //         break;
                //     }
                // }
            }
            // if e % 10000 == 0 {
            //     println!("Epoch: {}, Errors: {}", e, errors_this_epoch);
            // }
        }
    }
}