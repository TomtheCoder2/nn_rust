use nn_rust::ec::evolutionary_computation::EvolutionaryComputation;
use nn_rust::nn::data_set::DataSet;

fn main() {
    let seed = 123456;
    let generations = 100;
    let population_size = 100;
    let max_epochs = 1000;
    let max_epochs_start = 500;
    let max_calculations = 10000;
    let max_calculations_start = 10000;
    let max_layer_count = 5;
    let max_nodes = 1000;
    let max_nodes_start = 100;
    let max_learning_rate = 0.0001;
    let min_epochs = 20;
    let train_set: DataSet = DataSet::get_from_file("data.txt");
    let test_set: DataSet = DataSet::get_from_file("data.txt");
    let mut ec = EvolutionaryComputation::new(train_set, seed, generations, population_size, 4, 7, max_epochs, max_epochs_start, max_calculations, max_calculations_start, max_layer_count, max_nodes, max_nodes_start, max_learning_rate, min_epochs, test_set);
    println!("init done");
    ec.run();
    println!("finished");
}
