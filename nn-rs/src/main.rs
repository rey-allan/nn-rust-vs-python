use ndarray::{array, Array, Array2, Axis};
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use rand_pcg::Pcg64;
use std::io::Write;
use std::time::Instant;
use std::fs::File;

fn sigmoid(z: Array2<f64>) -> Array2<f64> {
    1. / (1. + (-z).mapv(f64::exp))
}

/// Structure of a Neural Network to learn the `XOR` function
/// - 2 input features
/// - 2 hidden neurons
/// - 1 output neuron
fn main() {
    // Generate dataset of `XOR` examples
    // We define the array as suggested by Andrew Ng: features are rows, examples are columns
    // This makes the computation of the linear combination match the math closer: W^T * X
    let x: Array2<f64> = array![[0., 0., 1., 1.], [0., 1., 0., 1.]];
    let y: Array2<f64> = array![[0., 1., 1., 0.]];
    let n = x.shape()[1] as f64;

    let seeds = vec![2, 10, 24, 45, 98, 120, 350, 600, 899, 1000];
    let mut runtimes: Vec<u128> = vec![];

    // Execute the training with each different seed and time them
    for seed in seeds {
        println!("Running with seed: {}", seed);
        let mut rng: Pcg64 = Pcg64::seed_from_u64(seed);
        let start = Instant::now();

        // Initialize weights and biases
        // Remember: they have to be initialized to random values in order to break symmetry!
        // Input to hidden
        let mut w0: Array2<f64> = Array::random_using((2, 2), Uniform::new(0., 1.), &mut rng);
        let mut b0: Array2<f64> = Array::random_using((2, 1), Uniform::new(0., 1.), &mut rng);
        // Hidden to output
        let mut w1: Array2<f64> = Array::random_using((2, 1), Uniform::new(0., 1.), &mut rng);
        let mut b1: Array2<f64> = Array::random_using((1, 1), Uniform::new(0., 1.), &mut rng);

        // Train using gradient descent
        let epochs = 100000;
        let alpha = 0.5;

        for _ in 0..epochs {
            // Forward propagation
            let a1 = sigmoid(w0.t().dot(&x) + &b0);
            let y_hat = sigmoid(w1.t().dot(&a1) + &b1);

            // Loss (MSE) multiplied by a 1/2 term to make the derivative easier
            let _loss = (1. / (2. * n)) * ((&y_hat - &y).mapv(|a| a.powi(2))).sum();

            // Backpropagation
            let dy_hat = (&y_hat - &y) / n;
            let dz2 = (&y_hat * (1. - &y_hat)) * &dy_hat;
            let dw1 = a1.dot(&dz2.t());
            // `ndarray` doesn't provide a way to keep dimensions so we need to reshape
            let db1 = Array::from_shape_vec((1, 1), dz2.sum_axis(Axis(1)).to_vec()).unwrap();
            let dz1 = (&a1 * (1. - &a1)) * (w1.dot(&(&dy_hat * (&y_hat * (1. - &y_hat)))));
            let dw0 = x.dot(&dz1.t());
            let db0 = Array::from_shape_vec((2, 1), dz1.sum_axis(Axis(1)).to_vec()).unwrap();

            // Weight and bias update
            // We average the gradients so that we can use a larger learning rate
            w0 = &w0 - alpha * (dw0 / n);
            w1 = &w1 - alpha * (dw1 / n);
            b0 = &b0 - alpha * (db0 / n);
            b1 = &b1 - alpha * (db1 / n);
        }

        // Record the time
        runtimes.push(start.elapsed().as_millis());
    }

    // Save runtimes to a file for further processing
    println!("Saving runtimes to `rust.txt`");
    let mut f = File::create("rust.txt").unwrap();
    for t in &runtimes {
        writeln!(f, "{}", t).expect("Failed to write runtime");
    }
}
