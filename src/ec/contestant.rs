use std::fmt;
use crate::ec::evolutionary_computation::{INPUT_SIZE, OUTPUT_SIZE, TEST_SET, TRAINING_SET};
use crate::nn::neural_network::NeuralNetwork;

#[derive(Debug, Clone)]
pub struct Contestant {
    pub layer_count: i32,
    pub layer_sizes: Vec<i32>,
    pub layers: Vec<i32>,
    pub lookup_table: Vec<i32>,
    pub neural_network: NeuralNetwork,
    pub calculations: i32,
    pub epochs: i32,
    pub learning_rate: f64,
    pub accuracy: f64,
    pub average_error: f64,
    pub max_error: f64,
    pub max_error_index: usize,
    pub cost: f64,
    pub scaled_cost: f64,
    pub fitness: f64,
    pub seed: i32,
    pub is_training: bool,
}

impl Contestant {
    pub fn new(epochs: i32, seed: i32, layer_sizes: Vec<i32>, learning_rate: f64) -> Contestant {
        let all_representations: (Vec<i32>, Vec<i32>) = Contestant::from_layer_sizes(layer_sizes.clone());
        let layers: Vec<i32>;
        unsafe { layers = Contestant::add_io(all_representations.1.clone(), INPUT_SIZE, OUTPUT_SIZE); }
        Contestant {
            layer_count: layer_sizes.len() as i32,
            layer_sizes: layer_sizes.clone(),
            layers: layers.clone(),
            lookup_table: all_representations.1,
            neural_network: NeuralNetwork::new(layers.iter().map(|x| *x as usize).collect(), learning_rate, seed),
            calculations: Contestant::calculations_calculator(layers.clone()),
            epochs,
            learning_rate,
            accuracy: 0.0,
            average_error: 0.0,
            max_error: 0.0,
            max_error_index: 0,
            cost: 0.0,
            scaled_cost: 0.0,
            fitness: 0.0,
            seed,
            is_training: false,
        }
    }

    pub fn add_io(layers: Vec<i32>, input_size: i32, output_size: i32) -> Vec<i32> {
        let mut new_layers = Vec::new();
        new_layers.push(input_size);
        for i in 0..layers.len() {
            new_layers.push(layers[i]);
        }
        new_layers.push(output_size);
        new_layers
    }

    pub fn calculations_calculator(layer_sizes: Vec<i32>) -> i32 {
        let mut calculations = 0;
        for i in 1..layer_sizes.len() {
            calculations += layer_sizes[i] * layer_sizes[i - 1];
            calculations += layer_sizes[i];
        }
        calculations
    }

    pub fn from_layer_sizes(mut layer_sizes: Vec<i32>) -> (Vec<i32>, Vec<i32>) {
        let mut layers = Vec::new();
        let mut lookup_table = Vec::new();
        for i in 0..layer_sizes.len() {
            if layer_sizes[i] != 0 {
                if layer_sizes[i] < 7 {
                    layer_sizes[i] = 7;
                }
                layers.push(layer_sizes[i]);
                lookup_table.push(i as i32);
            }
        }
        (lookup_table, layers)
    }


    pub fn fit(&mut self, _iter: i32) {
        unsafe {
            self.neural_network.fit(&TRAINING_SET.clone().unwrap().inputs, &TRAINING_SET.clone().unwrap().targets, self.epochs);
            // test the neural network
            // TODO: fix this part, cause idk how the java code worked and its ugly anyways
            let mut errors_per_color: Vec<Vec<f64>> = vec![vec![0.0; TEST_SET.clone().unwrap().targets[0].len()]; TEST_SET.clone().unwrap().inputs.len()];
            //println!("layer_sizes: {:?}", self.neural_network.layer_sizes);
            //println!("learning_rate: {:?}", self.neural_network.learning_rate);

            let mut predictions: Vec<usize> = vec![0; TEST_SET.clone().unwrap().inputs.len()];
            let mut outputs: Vec<Vec<f64>> = vec![vec![0.0; TEST_SET.clone().unwrap().targets[0].len()]; TEST_SET.clone().unwrap().inputs.len()];
            for i in 0..TEST_SET.clone().unwrap().inputs.len() {
                let output = self.neural_network.predict(TEST_SET.clone().unwrap().inputs[i].clone());
                outputs[i] = output.clone();
                //println!("output: {:?}", output);
                let target = TEST_SET.clone().unwrap().targets[i].clone();
                let mut correct_color= 0;
                let mut predicted_color= 0;
                for j in 0 .. output.len() {
                    if target[j] == 1.0 {
                        correct_color = j;
                    }
                    if output[j] > output[predicted_color] {
                        predicted_color = j;
                    }
                }
                if correct_color != predicted_color {
                    errors_per_color[i][correct_color] += 1.0;
                }
                predictions[i] = predicted_color;
            }
            println!("predictions: {:?}", predictions);
            println!("outputs: {:?}", outputs);
            /*
            println!("errors_per_color:");
            println!("{:?}", errors_per_color);

            for i in errors_per_color.clone() {
                println!("{:?}", i);
            }*/
            let mut errors_per_color_sum: Vec<f64> = vec![0.0; TEST_SET.clone().unwrap().targets[0].len()];
            for i in 0..errors_per_color.len() {
                for j in 0..errors_per_color[i].len() {
                    errors_per_color_sum[j] += errors_per_color[i][j];
                }
            }
            //println!("errors_per_color_sum: {:?}", errors_per_color_sum);
            let (max_error_index, max_error) = errors_per_color_sum.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
            let average_error = errors_per_color_sum.iter().sum::<f64>() / errors_per_color_sum.len() as f64;
            let accuracy = 0.0; //tmp, TODO: add accuracy functionality here
            //println!("max_error: {}, max_error_index: {}, average_error: {}, accuracy: {}", max_error, max_error_index, average_error, accuracy);
            self.accuracy = accuracy;
            self.average_error = average_error / TEST_SET.clone().unwrap().inputs.len() as f64 * 100.0;
            self.max_error = *max_error / TEST_SET.clone().unwrap().inputs.len() as f64 * 100.0;
            self.max_error_index = max_error_index;
            self.cost = average_error + max_error;
        }
        // println!("finished: {}", iter);
    }

    pub fn print_properties(&self) {
        println!("Contestant Properties:");
        println!("\tlayer_count: {}", self.layer_count);
        println!("\tlayer_sizes: {:?}", self.layer_sizes);
        println!("\tlayers: {:?}", self.layers);
        println!("\tlookup_table: {:?}", self.lookup_table);
        println!("\tcalculations: {}", self.calculations);
        println!("\tepochs: {}", self.epochs);
        println!("\tlearning_rate: {}", self.learning_rate);
        println!("\taccuracy: {}", self.accuracy);
        println!("\taverage_error: {}", self.average_error);
        println!("\tmax_error: {}", self.max_error);
        println!("\tcost: {}", self.cost);
        println!("\tscaled_cost: {}", self.scaled_cost);
        println!("\tfitness: {}", self.fitness);
        println!("\tseed: {}", self.seed);
        println!("\tis_training: {}", self.is_training);
    }
}

impl fmt::Display for Contestant {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Contestant {{layer_count: {}\n layer_sizes: {:?}\n layers: {:?}\n lookup_table: {:?}\n calculations: {}\n epochs: {}\n learning_rate: {}\n accuracy: {}\n average_error: {}\n max_error: {}\n cost: {}\n scaled_cost: {}\n fitness: {}\n seed: {}\n is_training: {} }}", self.layer_count, self.layer_sizes, self.layers, self.lookup_table, self.calculations, self.epochs, self.learning_rate, self.accuracy, self.average_error, self.max_error, self.cost, self.scaled_cost, self.fitness, self.seed, self.is_training)
    }
}