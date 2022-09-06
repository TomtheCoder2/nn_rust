use std::fs::File;
use std::io::Read;

#[derive(Debug, Clone)]
pub struct DataSet {
    pub inputs: Vec<Vec<f64>>,
    pub targets: Vec<Vec<f64>>,
}

impl DataSet {
    pub fn new(inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>) -> DataSet {
        DataSet { inputs, targets }
    }

    pub fn new_empty() -> DataSet {
        DataSet { inputs: Vec::new(), targets: Vec::new() }
    }

    pub fn get_from_file(file_name: &str) -> DataSet {
        let mut file = File::open(file_name).expect("File not found");
        let mut contents = String::new();
        file.read_to_string(&mut contents).expect("Something went wrong reading the file");
        // format: {input1.1, input1.2, input1.3, input1.4}, {target1.1, target1.2...}
        let inputs: Vec<Vec<f64>> = contents
            .split("},{")
            .map(|x| x.replace("{", "").replace("}", ""))
            .map(|x| x.split(",").map(|x| x.replace(" ", "").parse::<f64>().unwrap()).collect())
            .collect();
        let mut targets: Vec<Vec<f64>> = Vec::new();
        for i in 0..7 {
            for __ in 0..inputs.len() / 7 {
                let mut tar: Vec<f64> = (0..7).map(|_| 0.0).collect();
                tar[i] = 1.0;
                targets.push(tar);
            }
        };
        DataSet { inputs, targets }
    }
}