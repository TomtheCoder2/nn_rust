use crate::ec::EvolutionaryComputation::{INPUT_SIZE, OUTPUT_SIZE, TEST_SET, TRAINING_SET};
use crate::nn::data_set::DataSet;
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
    pub cost: f64,
    pub scaled_cost: f64,
    pub fitness: f64,
    pub seed: i32,
    pub is_training: bool,
}

impl Contestant {
    pub fn new(epochs: i32, seed: i32, layer_sizes: Vec<i32>, learning_rate: f64) -> Contestant {
        let all_representations: (Vec<i32>, Vec<i32>) = Contestant::from_layer_sizes(layer_sizes.clone());
        let mut layers: Vec<i32>;
        unsafe { layers = Contestant::add_io(all_representations.1.clone(), INPUT_SIZE, OUTPUT_SIZE); }
        Contestant {
            layer_count: layer_sizes.len() as i32,
            layer_sizes: layer_sizes.clone(),
            layers: layers.clone(),
            lookup_table: all_representations.1,
            neural_network: NeuralNetwork::new(layers.clone().iter().map(|x| *x as usize).collect(), learning_rate, seed),
            calculations: Contestant::calculations_calculator(layers.clone()),
            epochs,
            learning_rate,
            accuracy: 0.0,
            average_error: 0.0,
            max_error: 0.0,
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


    pub fn fit(&mut self, iter: i32) {
        unsafe {
            self.neural_network.fit(&TRAINING_SET.clone().unwrap().inputs.clone(), &TRAINING_SET.clone().unwrap().targets.clone(), self.epochs);
            // test the neural network
            // TODO: fix this part, cause idk how the java code worked and its ugly anyways
            let (accuracy, mut average_error, max_error) = (0.0, 0.0, 0.0);
            for i in 0..TEST_SET.clone().unwrap().inputs.len() {
                let output = self.neural_network.predict(TEST_SET.clone().unwrap().inputs[i].clone());
                let target = TEST_SET.clone().unwrap().targets[i].clone();
                let bit_output: Vec<f64> = output.iter().map(|x| if *x > 0.5 { 1.0 } else { 0.0 }).collect();
                if bit_output != target {
                    average_error += 1.0;
                }
            }
            average_error /= TEST_SET.clone().unwrap().inputs.len() as f64;
            self.accuracy = accuracy;
            self.cost = average_error;
        }
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