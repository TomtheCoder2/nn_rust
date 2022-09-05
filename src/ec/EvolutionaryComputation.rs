use crate::nn::data_set::DataSet;

const COST_SCALER: f64 = 1.2;
const ERROR_SCALER: f64 = 2.5;
const ACCURACY_SCALER: f64 = 1.2;
const CALCULATIONS_SCALER: f64 = 0.2;
const EPOCHS_SCALER: f64 = 0.2;
const PARAMETER_CHANGE_RATE_EXP: i32 = 5;
const PARAMETER_CHANGE_RATE_LINEAR: i32 = 1;
const EXP_PARAMETER_CHANGE_RATE_EXP: i32 = 5;
const EXP_PARAMETER_CHANGE_RATE_LINEAR: i32 = 10;
const ASEXUAL_REPRODUCTION: f64 = 0.75;
const SEXUAL_REPRODUCTION: f64 = 0.22;

//asexual reproduction
//(-1 .. 1 rand) ** change_rate * (max - min) + min
public final static double asexual_keep_rate = 0.95;
//sexual reproduction
//how much to keep from both parents - the rest is mixed together
public final static double sexual_keep_rate = 0.3;

pub struct EvolutionaryComputation {
    population: Vec<Contestant>,
    current_generation: i32,
    population_size: i32,
    input_size: i32,
    output_size: i32,
    max_epochs: i32,
    max_epochs_start: i32,
    min_epochs: i32,
    max_calculations: i32,
    max_calculations_start: i32,
    max_layer_count: i32,
    max_nodes: i32,
    max_nodes_start: i32,
    max_learning_rate: i32,
    generations: i32,
    training_data: DataSet,
    test_data: DataSet,
    seed: i32,
}

impl EvolutionaryComputation {
    pub fn new(training_set: DataSet, seed: i32, generations: i32, population_size: i32, input_size: i32, output_size: i32, max_epochs: i32, max_epochs_start: i32, max_calculations: i32, max_calculations_start: i32,max_layer_count:i32,  max_nodes: i32, max_nodes_start: i32, max_learning_rate: i32, min_epochs: i32, test_set: DataSet) -> EvolutionaryComputation {
        EvolutionaryComputation {
            population: Vec::new(),
            current_generation: 0,
            population_size,
            input_size,
            output_size,
            max_epochs,
            max_epochs_start,
            min_epochs,
            max_calculations,
            max_calculations_start,
            max_layer_count,
            max_nodes,
            max_nodes_start,
            max_learning_rate,
            generations,
            training_data: training_set,
            test_data: test_set,
            seed,
        }
    }
    pub fn getBest(population: &Vec<Contestant>) -> Contestant {
        let mut best = population[0];
        for i in 1..population.len() {
            if population[i].cost > best.cost {
                best = population[i];
            }
        }
        best.clone()
    }
    pub fn getWorst(population: &Vec<Contestant>) -> Contestant {
        let mut worst = population[0];
        for i in 1..population.len() {
            if population[i].cost < worst.cost {
                worst = population[i];
            }
        }
        worst.clone()
    }

    pub fn run(&mut self) {
        let new_population:Vec<Contestant> = Vec::new();
        asdf
    }
}