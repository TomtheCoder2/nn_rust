use crate::nn::data_set::DataSet;
use crate::nn::neural_network::NeuralNetwork;

struct Contestant {
    layer_count: i32,
    layer_sizes: Vec<i32>,
    layers: Vec<i32>,
    lookup_table: Vec<i32>,
    training_set: DataSet,
    test_set: DataSet,
    neural_network: NeuralNetwork,
    calculations: i32,
    epochs: i32,
    learning_rate: f64,
    accuracy: f64,
    average_error: f64,
    max_error: f64,
    cost: f64,
    scaled_cost: f64,
    fitness: f64,
    seed: i32,
    is_training: bool,
}

impl Contestant {
    pub fn new(training_set: DataSet, epochs: i32, seed: i32, layer_sizes: Vec<i32>, learning_rate: f64, test_set: DataSet, input_size: i32, output_size: i32) -> Contestant {
        let all_representations: (Vec<i32>, Vec<i32>) = Contestant::from_layer_sizes(layer_sizes);
        let layers = Contestant::add_io(all_representations.second, input_size, output_size);
        Contestant {
            layer_count: layer_sizes.len() as i32,
            layer_sizes: layer_sizes.clone(),
            layers,
            lookup_table: all_representations.1,
            training_set,
            test_set,
            neural_network: NeuralNetwork::new(layers.clone().iter().map(|x| x as usize).collect(), learning_rate, seed),
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

    pub fn from_layer_sizes(layer_sizes: Vec<i32>) -> (Vec<i32>, Vec<i32>) {
        let mut layers = Vec::new();
        let mut lookup_table = Vec::new();
        for i in 0..layer_sizes.len() {
            if layer_sizes[i] != 0 {
                if layer_sizes[i] < 7 {
                    layer_sizes[i] = 7;
                }
                layers.push(layer_sizes[i]);
                lookup_table.push(layer_count);
            }
        }
        (layers, lookup_table)
    }

    pub fn fit(&mut self, iter: i32) {
        self.neural_network.fit(&self.training_set.inputs, &self.training_set.targets, self.epochs);
        // test the neural network
        let  (accuracy, mut average_error, max_error) = (0.0, 0.0, 0.0);
        for i in 0..self.test_set.inputs.len() {
            let output = self.neural_network.predict(self.test_set.inputs[i].clone());
            let target = &self.test_set.targets[i];
            let bit_output = output.iter().map(|x| if *x > 0.5 { 1 } else { 0 }).collect::<Vec<i32>>();
            if bit_output != target {
                average_error += 1.0;
            }
        }
        average_error /= self.test_set.inputs.len() as f64;
    }
}