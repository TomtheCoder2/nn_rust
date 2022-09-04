mod matrix;
mod neural_network;
mod data_set;

use crate::data_set::DataSet;
use crate::neural_network::{NeuralNetwork};


fn main() {
    let mut nn = NeuralNetwork::new(vec![4, 74, 89, 7], 0.000056, 1);
    let ds = DataSet::getFromFile("data.txt");
    // print data_set
    nn.fit(&ds.inputs, &ds.targets, 10);
}