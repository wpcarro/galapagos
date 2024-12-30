//! # Galapagos
//!
//! Low-dependency, simple evolutionary solver.

use rand::{self, Rng};

#[derive(PartialEq)]
pub enum Goal {
    Maximize,
    Minimize,
}

/// A single solution attempt.
#[derive(Debug, Clone)]
pub struct Individual {
    /// The values this individual has set for its sliders. Some people refer to
    /// these as "genes".
    pub values: Vec<f32>,
    /// Cached result of calling `f` with this individual's `values`.
    pub fitness: f32,
}

/// The near-optimal answer for a given series of evolutions.
#[derive(Debug)]
pub struct Solution {
    /// The most fit individual from a population that evolved over many
    /// generations.
    pub champion: Individual,
    /// Optionally store the history of each generation's most fit individual.
    /// Useful to plot the behavior for debugging.
    pub history: Option<Vec<(f32, f32)>>,
}

/// Solver configuration.
#[derive(Debug)]
pub struct Config {
    /// The number of individuals to evaluate in each generation.
    pub population_size: u32,
    /// The number of evolution iterations to run.
    pub generation_count: u32,
    /// The likelihood of any two "parents" of mixing their "genes" instead of
    /// passing them on directly.
    pub crossover_rate: f32,
    /// The likelihood of any one gene changing in value.
    pub mutation_rate: f32,
    /// Whether or not to store a vector recording the fitness of each
    /// generation's most fit individual.
    pub record_history: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            population_size: 200,
            generation_count: 1_000,
            crossover_rate: 0.70,
            mutation_rate: 0.10,
            record_history: false,
        }
    }
}

/// Attempts to find a near-optimal solution for a given equation.
///
/// # Arguments
///
/// * `f` - The function to solve.
/// * `sliders` - List of tuples representing min, max bounds for a "slider"
/// that the solver can modulate when searching for solutions.
/// * `goal` - Whether to search for a maximal or minimal solution.
/// * `cfg` - Solver configuration.
///
/// # Examples
///
/// ```
/// use galapagos::{solve, Goal};
/// let solution = solve(
///     |xs| {
///         // Sphere function
///         let x = xs[0];
///         let y = xs[1];
///         let z = xs[2];
///         x.powi(2) + y.powi(2) + z.powi(2)
///     },
///     vec![(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)],
///     Goal::Minimize,
///     Default::default(),
/// );
/// assert_eq!(solution.champion.values.len(), 3);
/// ```
pub fn solve<F: Fn(Vec<f32>) -> f32>(
    f: F,
    sliders: Vec<(f32, f32)>,
    goal: Goal,
    cfg: Config,
) -> Solution {
    assert!(
        cfg.population_size > 2,
        "Population size should be at least two"
    );
    assert!(
        cfg.population_size % 2 == 0,
        "We only support even-numbered population size at the moment"
    );
    assert!(cfg.generation_count > 1, "Need at least one generation");

    let evaluate_fitness = |idv: &mut Individual| -> () {
        idv.fitness = f(idv.values.clone());
    };

    // Function that returns the most fitness individual from a list of
    // candidates.
    let compete = |xs: Vec<Individual>| -> Individual {
        assert!(xs.len() >= 2);
        match goal {
            Goal::Maximize => xs
                .iter()
                .max_by(|x, y| x.fitness.partial_cmp(&y.fitness).unwrap())
                .unwrap()
                .clone(),
            Goal::Minimize => xs
                .iter()
                .min_by(|x, y| x.fitness.partial_cmp(&y.fitness).unwrap())
                .unwrap()
                .clone(),
        }
    };

    let mut rng = rand::thread_rng();

    // population size (1,000)
    let mut parents: Vec<Individual> = Vec::with_capacity(cfg.population_size as usize);
    for _ in 0..cfg.population_size {
        let mut xs = Vec::with_capacity(sliders.len());
        for j in 0..sliders.len() {
            xs.push(rng.gen_range(sliders[j].0..=sliders[j].1));
        }
        let mut i = Individual {
            values: xs,
            fitness: 0.0,
        };
        evaluate_fitness(&mut i);
        parents.push(i);
    }

    let mut history: Vec<(f32, f32)> = if cfg.record_history {
        Vec::with_capacity(cfg.generation_count as usize)
    } else {
        Vec::with_capacity(0)
    };

    for generation in 0..cfg.generation_count as usize {
        // host a tournament
        let mut children: Vec<Individual> = Vec::with_capacity(cfg.population_size as usize);
        for _ in 0..cfg.population_size {
            let i = rng.gen_range(0..cfg.population_size as usize);
            let j = rng.gen_range(0..cfg.population_size as usize);
            let lhs = &parents[i];
            let rhs = &parents[j];
            let winner = compete(vec![lhs.clone(), rhs.clone()]);
            children.push(winner);
        }
        let champion = compete(parents);

        if cfg.record_history {
            history.push((generation as f32, champion.fitness))
        }

        // crossover the winners
        for i in 0..(cfg.population_size as usize / 2) {
            // crossover rate
            if rng.gen::<f32>() < cfg.crossover_rate {
                let mut mom = children[i * 2].clone();
                let mut dad = children[i * 2 + 1].clone();

                let point = rng.gen_range(0..sliders.len());
                for j in 0..sliders.len() {
                    mom.values[j] = if j < point {
                        mom.values[j]
                    } else {
                        dad.values[j]
                    };
                    dad.values[j] = if j < point {
                        dad.values[j]
                    } else {
                        mom.values[j]
                    };
                }

                children[i * 2] = mom;
                children[i * 2 + 1] = dad;
            }
        }

        // mutate the winners
        for i in 0..cfg.population_size as usize {
            let x = &mut children[i];
            for j in 0..sliders.len() {
                if rng.gen::<f32>() < cfg.mutation_rate {
                    x.values[j] =
                        (x.values[j] + rng.gen_range(-0.5..=0.5)).clamp(sliders[j].0, sliders[j].1);
                }
            }
            evaluate_fitness(x);
        }
        parents = children;
    }
    let champion = compete(parents);
    Solution {
        champion,
        history: if cfg.record_history {
            Some(history)
        } else {
            None
        },
    }
}
