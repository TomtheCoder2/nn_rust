use pgfplots::axis::plot::Plot2D;

fn main() {
    let mut plot = Plot2D::new();
    plot.coordinates = (-100..100)
        .into_iter()
        .map(|i| (f64::from(i), f64::from(i*i)).into())
        .collect();

    plot.show()?;
}
