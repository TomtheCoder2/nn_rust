use nn_rust::nn::neural_network::NeuralNetwork;
use nn_rust::nn::data_set::DataSet;
use std::time::{Duration, Instant};


fn main() {
    let mut nn = NeuralNetwork::new(vec![4, 74, 89, 7], 0.000056, 1);
    let ds = DataSet::getFromFile("data.txt");
    // print data_set
    let start = Instant::now();
    nn.fit(&ds.inputs, &ds.targets, 100);
    let duration = start.elapsed();
    println!("Time elapsed in nn.fit() is: {:?}", duration);
}