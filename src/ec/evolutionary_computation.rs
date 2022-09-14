use std::thread;
use rand::{Rng, thread_rng};
use crate::ec::contestant::Contestant;
use crate::nn::data_set::DataSet;

const COST_SCALER: f64 = 1.2;
// const ERROR_SCALER: f64 = 2.5;
// const ACCURACY_SCALER: f64 = 1.2;
// const CALCULATIONS_SCALER: f64 = 0.2;
// const EPOCHS_SCALER: f64 = 0.2;
// const PARAMETER_CHANGE_RATE_EXP: i32 = 5;
// const PARAMETER_CHANGE_RATE_LINEAR: i32 = 1;
const EXP_PARAMETER_CHANGE_RATE_EXP: i32 = 5;
const EXP_PARAMETER_CHANGE_RATE_LINEAR: i32 = 10;
const ASEXUAL_REPRODUCTION: f64 = 0.75;
const SEXUAL_REPRODUCTION: f64 = 0.22;

//asexual reproduction
//(-1 .. 1 rand) ** change_rate * (max - min) + min
const ASEXUAL_KEEP_RATE: f64 = 0.95;
//sexual reproduction
//how much to keep from both parents - the rest is mixed together
const SEXUAL_KEEP_RATE: f64 = 0.3;

pub static mut TRAINING_SET: Option<DataSet> = None;
pub static mut TEST_SET: Option<DataSet> = None;
pub static mut OUTPUT_SIZE: i32 = 0;
pub static mut INPUT_SIZE: i32 = 0;

pub struct EvolutionaryComputation {
    population: Vec<Contestant>,
    current_generation: i32,
    population_size: i32,
    max_epochs: i32,
    max_epochs_start: i32,
    min_epochs: i32,
    max_calculations: i32,
    max_calculations_start: i32,
    max_layer_count: i32,
    max_nodes: i32,
    max_nodes_start: i32,
    max_learning_rate: f64,
    generations: i32,
    seed: i32,
}

impl EvolutionaryComputation {
    pub fn new(training_set: DataSet, seed: i32, generations: i32, population_size: i32, input_size: i32, output_size: i32, max_epochs: i32, max_epochs_start: i32, max_calculations: i32, max_calculations_start: i32, max_layer_count: i32, max_nodes: i32, max_nodes_start: i32, max_learning_rate: f64, min_epochs: i32, test_set: DataSet) -> EvolutionaryComputation {
        unsafe {
            TRAINING_SET = Option::from(training_set);
            TEST_SET = Option::from(test_set);
            INPUT_SIZE = input_size;
            OUTPUT_SIZE = output_size;
        }
        EvolutionaryComputation {
            population: Vec::new(),
            current_generation: 0,
            population_size,
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
            seed,
        }
    }
    pub fn get_best(population: &Vec<Contestant>) -> Contestant {
        let mut best = population[0].clone();
        for i in 1..population.len() {
            if population[i].fitness > best.fitness {
                best = population[i].clone();
            }
        }
        best
    }
    pub fn get_worst(population: &Vec<Contestant>) -> Contestant {
        let mut worst = 0;
        for i in 1..population.len() {
            if population[i].cost < population[worst].cost {
                worst = i
            }
        }
        population[worst].clone()
    }

    pub fn run(&mut self) {
        let mut new_population: Vec<Contestant>;
        for _i in 0..self.population_size {
            self.population.push(self.generate_random_start());
        }
        for i in 0..self.generations as usize {
            println!("Start Gen #{}", i);
            let start = std::time::Instant::now();
            self.current_generation = i as i32;
            // TODO: thread magic
            let mut todo = vec![];
            let mut pop = self.population.clone();
            for j in 0..self.population_size as usize {
                let iter = j; //mb needs a .clone()
                let mut pop_j = pop[j].clone();
                todo.push(thread::spawn(move || {
                    pop_j.fit(iter as i32);
                    println!("thread finished {}: error: {}|{}, epochs: {}, layers: {:?}", iter, pop_j.average_error as i32, pop_j.max_error as i32, pop_j.epochs, pop_j.layers);
                    (pop_j, iter)
                }));
            }
            for t in todo {
                let res = t.join().unwrap();
                pop[res.1 as usize] = res.0;
            }
            self.population = pop;
            // stop timer
            let duration = start.elapsed();
            println!("Time elapsed Gen {} is: {:?}", i, duration);
            println!("Generation: {}", i);
            println!("Best: {}", EvolutionaryComputation::get_best(&self.population).average_error);
            println!("Worst: {}", EvolutionaryComputation::get_worst(&self.population).average_error);
            println!("Best Contestant: \n {}", EvolutionaryComputation::get_best(&self.population));
            new_population = self.next_gen(&mut self.population.clone());
            self.population = new_population;
        }
    }

    pub fn generate_random_start(&self) -> Contestant {
        let mut layer_sizes_: Vec<i32> = Vec::new();
        for _i in 0..self.max_layer_count {
            layer_sizes_.push(thread_rng().gen_range(0..self.max_nodes_start));
        }
        unsafe {
            let from_layer = Contestant::from_layer_sizes(layer_sizes_.clone());
            while Contestant::calculations_calculator(Contestant::add_io(Contestant::from_layer_sizes(layer_sizes_.clone()).1, INPUT_SIZE, OUTPUT_SIZE)) > self.max_calculations_start {
                layer_sizes_[from_layer.0[thread_rng().gen_range(0..from_layer.0.len() - 1)] as usize] = 0;
            }
        }
        let lookup: Vec<i32> = Contestant::from_layer_sizes(layer_sizes_.clone()).0;
        let keep_rate: f64 = thread_rng().gen_range(0.0..1.0);

        for i in lookup {
            if thread_rng().gen_range(0.0..1.0) < keep_rate {
                layer_sizes_[i as usize] = 0;
            }
        }
        Contestant::new(thread_rng().gen_range(self.min_epochs..self.max_epochs_start),
                        self.seed, layer_sizes_, thread_rng().gen_range(0.0..self.max_learning_rate))
    }

    pub fn next_gen(&self, population: &mut Vec<Contestant>) -> Vec<Contestant> {
        // println!("Next Gen");
        let mut next_population: Vec<Contestant> = Vec::new();
        for i in 0..self.population_size as usize {
            population[i].scaled_cost = population[i].cost.powf(COST_SCALER);
        }
        let worst_contestant = EvolutionaryComputation::get_worst(&population.clone());
        let mut fitness_sum: f64 = 0.0;
        for i in 0..self.population_size as usize {
            population[i].fitness = worst_contestant.scaled_cost / population[i].scaled_cost;
            fitness_sum += population[i].fitness;
        }
        // for i in 0..self.population_size as usize {
        //     population[i].print_properties();
        // }
        for _i in 0..population.len() {
            // println!("pop len: {}", population.len());
            let point = self.contestant_int(population, fitness_sum);
            let r_number: f64 = thread_rng().gen_range(0.0..1.0);
            if r_number < ASEXUAL_REPRODUCTION {
                next_population.push(self.mutate(&mut population[point].clone()));
            } else if r_number < SEXUAL_REPRODUCTION + ASEXUAL_REPRODUCTION {
                let point2 = self.contestant_int(population, fitness_sum);
                next_population.push(self.mutate(&mut self.sexual_reproduction(&population[point], &population[point2])));
            } else {
                next_population.push(self.generate_random_start());
            }
        }
        next_population
    }

    pub fn contestant_int(&self, population: &Vec<Contestant>, fitness_sum: f64) -> usize {
        let mut current_sum = population[0].fitness;
        let mut point = 0;
        let goal = thread_rng().gen_range(0.0..fitness_sum);
        while current_sum < goal {
            point += 1;
            current_sum += population[point].fitness;
        }
        point
    }

    pub fn sexual_reproduction(&self, a: &Contestant, b: &Contestant) -> Contestant {
        let mut layer_sizes: Vec<i32> = Vec::new();
        for i in 0..self.max_layer_count {
            if thread_rng().gen_range(0.0..1.0) < 0.5 {
                layer_sizes.push(a.layer_sizes[i as usize]);
            } else {
                layer_sizes.push(b.layer_sizes[i as usize]);
            }
        }
        Contestant::new(EvolutionaryComputation::merge(a.epochs as f64, b.epochs as f64) as i32, EvolutionaryComputation::merge(a.seed as f64, b.seed as f64) as i32, layer_sizes, EvolutionaryComputation::merge(a.learning_rate, b.learning_rate))
    }

    pub fn merge(a: f64, b: f64) -> f64 {
        if thread_rng().gen_range(0.0..1.0) < SEXUAL_KEEP_RATE {
            if thread_rng().gen_range(0.0..1.0) < 0.5 {
                a
            } else {
                b
            }
        } else {
            (a + b) / 2.0
        }
    }

    pub fn mutate(&self, contestant: &mut Contestant) -> Contestant {
        let mut layer_sizes: Vec<i32> = contestant.layer_sizes.clone();
        for i in 0..self.max_layer_count as usize {
            let r_number: f64 = thread_rng().gen_range(0.0..1.0);
            if layer_sizes[i] > 0 {
                if r_number < ASEXUAL_KEEP_RATE {
                    layer_sizes[i] = self.calculate_change(1.0, self.max_nodes as f64, layer_sizes[i as usize] as f64, contestant.fitness) as i32;
                } else {
                    layer_sizes[i] = 0;
                }
            } else if r_number < ASEXUAL_KEEP_RATE {
                layer_sizes[i] = thread_rng().gen_range(0..self.max_nodes_start) as i32;
            }
        }
        unsafe {
            let mut from_layer = Contestant::from_layer_sizes(layer_sizes.clone());
            while Contestant::calculations_calculator(Contestant::add_io(from_layer.1.clone(), INPUT_SIZE, OUTPUT_SIZE)) > self.max_calculations {
                let next_int = from_layer.0.len();
                // println!("next_int: {}", next_int);
                layer_sizes[from_layer.0[thread_rng().gen_range(0..next_int)] as usize] = 0;
                from_layer = Contestant::from_layer_sizes(layer_sizes.clone());
            }
        }

        Contestant::new(self.calculate_change(1.0, self.max_epochs as f64, contestant.epochs as f64, contestant.fitness) as i32,
                        self.seed,
                        layer_sizes,
                        self.calculate_change(0.0, self.max_learning_rate, contestant.learning_rate, contestant.fitness))
    }

    pub fn calculate_change(&self, min: f64, max: f64, current: f64, fitness: f64) -> f64 {
        // java code:
        //     return Math.min(
        //                 Math.max(
        //                         current + (
        //                                 (
        //                                         (Math.pow(((generator.nextDouble() * 2) - 1), exp_parameter_change_rate_exp)
        //                                                 * (current - min))
        //                                                 * exp_parameter_change_rate_linear
        // //                                                * (current / (max - min))
        //                                 ) / fitness),
        //                         min),
        //                 max);
        min.max(
            max.min(
                current + (
                    (
                        ((thread_rng().gen_range(0.0..1.0) * 2.0) - 1.0) as f64)
                        .powf(EXP_PARAMETER_CHANGE_RATE_EXP as f64)
                        * (current - min)
                        * EXP_PARAMETER_CHANGE_RATE_LINEAR as f64
                        / fitness
                )
            )
        )
    }
}
