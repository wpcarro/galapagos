//! # Galapagos
//!
//! Low-dependency, simple evolutionary solver.

use core::f32;

use rand::{self, Rng};

#[derive(PartialEq)]
pub enum Goal {
    Maximize,
    Minimize,
}

/// A single solution attempt.
#[derive(Debug, Clone, PartialEq)]
pub struct Individual {
    /// The values this individual has set for its sliders. Some people refer to
    /// these as "genes".
    pub values: Vec<f32>,
    /// Cached result of calling `f` with this individual's `values`.
    pub fitness: f32,
}

// Converting from an array of structs (AoS) to a struct of arrays (SoA)
#[derive(Debug)]
struct Individuals {
    num_genes: u32,
    values: Vec<f32>,
    fitness: Vec<f32>,
}

impl Individuals {
    fn with_capacity(num_genes: u32, capacity: u32) -> Individuals {
        Individuals {
            num_genes,
            values: Vec::with_capacity((num_genes * capacity) as usize),
            fitness: Vec::with_capacity(capacity as usize),
        }
    }

    fn get(&self, i: u32) -> Individual {
        let lo = (i * self.num_genes) as usize;
        let hi = lo + self.num_genes as usize;
        Individual {
            values: Vec::from(&self.values[lo..hi]),
            fitness: self.fitness[i as usize],
        }
    }

    fn set(&mut self, i: u32, x: Individual) -> () {
        let lo = (i * self.num_genes) as usize;
        for (j, value) in x.values.into_iter().enumerate() {
            self.values[lo + j] = value;
        }
        self.fitness[i as usize] = x.fitness;
    }

    fn push(&mut self, x: Individual) -> () {
        for value in x.values.into_iter() {
            self.values.push(value);
        }
        self.fitness.push(x.fitness);
    }

    fn len(&self) -> u32 {
        self.fitness.len() as u32
    }
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
    /// Whether or not to print a progress bar periodically.
    pub print_progress: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            population_size: 200,
            generation_count: 1_000,
            crossover_rate: 0.70,
            mutation_rate: 0.10,
            record_history: false,
            print_progress: false,
        }
    }
}

fn compete_two(goal: &Goal, lhs: &Individual, rhs: &Individual) -> Individual {
    match goal {
        Goal::Maximize => {
            if lhs.fitness > rhs.fitness {
                lhs.clone()
            } else {
                rhs.clone()
            }
        }
        Goal::Minimize => {
            if lhs.fitness < rhs.fitness {
                lhs.clone()
            } else {
                rhs.clone()
            }
        }
    }
}

fn compete_many(goal: &Goal, xs: &Individuals) -> Individual {
    assert!(xs.len() > 2);
    match goal {
        Goal::Maximize => {
            let mut i: u32 = 0;
            for j in 1..xs.len() {
                if xs.fitness[j as usize] > xs.fitness[i as usize] {
                    i = j;
                }
            }
            let lo = (i * xs.num_genes) as usize;
            let hi = lo + xs.num_genes as usize;
            Individual {
                values: Vec::from(&xs.values[lo..hi]),
                fitness: xs.fitness[i as usize],
            }
        }
        Goal::Minimize => {
            let mut i: u32 = 0;
            for j in 1..xs.len() {
                if xs.fitness[j as usize] < xs.fitness[i as usize] {
                    i = j;
                }
            }
            let lo = (i * xs.num_genes) as usize;
            let hi = lo + xs.num_genes as usize;
            Individual {
                values: Vec::from(&xs.values[lo..hi]),
                fitness: xs.fitness[i as usize],
            }
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
///     &[(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)],
///     Goal::Minimize,
///     Default::default(),
/// );
/// assert_eq!(solution.champion.values.len(), 3);
/// ```
pub fn solve<F: Fn(&[f32]) -> f32>(
    f: F,
    sliders: &[(f32, f32)],
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

    let mut rng = rand::thread_rng();

    let mut parents: Individuals =
        Individuals::with_capacity(sliders.len() as u32, cfg.population_size);
    for _ in 0..cfg.population_size {
        let mut xs = Vec::with_capacity(sliders.len());
        for j in 0..sliders.len() {
            xs.push(rng.gen_range(sliders[j].0..=sliders[j].1));
        }
        let mut i = Individual {
            values: xs,
            fitness: 0.0,
        };
        i.fitness = f(&i.values);
        parents.push(i);
    }

    let mut history: Vec<(f32, f32)> = if cfg.record_history {
        Vec::with_capacity(cfg.generation_count as usize)
    } else {
        Vec::with_capacity(0)
    };

    // Cache the optimum that we've encountered in case we meander away from it.
    // There is no guarantee that the most fit individual in the last generation
    // is the most fit individual we observed.
    let mut global_champion: Individual = match goal {
        Goal::Maximize => Individual {
            values: Vec::with_capacity(sliders.len()),
            fitness: f32::MIN,
        },
        Goal::Minimize => Individual {
            values: Vec::with_capacity(sliders.len()),
            fitness: f32::MAX,
        },
    };
    for generation in 0..cfg.generation_count as usize {
        if cfg.print_progress && rng.gen::<f32>() < 0.10 {
            println!(
                "Loading... {:.1}%",
                generation as f32 / cfg.generation_count as f32 * 100.0
            );
        }
        // host a tournament
        let mut children: Individuals =
            Individuals::with_capacity(sliders.len() as u32, cfg.population_size);
        for _ in 0..cfg.population_size {
            let i = rng.gen_range(0..cfg.population_size as usize);
            let j = rng.gen_range(0..cfg.population_size as usize);
            let lhs = &parents.get(i as u32);
            let rhs = &parents.get(j as u32);
            let winner = compete_two(&goal, lhs, rhs);
            children.push(winner);
        }
        let local_champion = compete_many(&goal, &parents);
        global_champion = compete_two(&goal, &local_champion, &global_champion);

        if cfg.record_history {
            history.push((generation as f32, local_champion.fitness))
        }

        // crossover the winners
        for i in 0..(cfg.population_size as usize / 2) {
            // crossover rate
            if rng.gen::<f32>() < cfg.crossover_rate {
                let mut mom = children.get(i as u32 * 2).clone();
                let mut dad = children.get(i as u32 * 2 + 1).clone();

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

                children.set(i as u32 * 2, mom);
                children.set(i as u32 * 2 + 1, dad);
            }
        }

        // mutate the winners
        for i in 0..cfg.population_size as usize {
            let mut x = children.get(i as u32);
            for j in 0..sliders.len() {
                if rng.gen::<f32>() < cfg.mutation_rate {
                    // Nudge +/-10% of the slider.
                    let (min, max) = (sliders[j].0, sliders[j].1);
                    let nudge = 0.10 * (max - min);
                    x.values[j] = (x.values[j] + rng.gen_range(-nudge..=nudge)).clamp(min, max);
                }
            }
            x.fitness = f(&x.values);
            children.set(i as u32, x);
        }
        parents = children;
    }
    let local_champion = compete_many(&goal, &parents);
    global_champion = compete_two(&goal, &local_champion, &global_champion);
    Solution {
        champion: global_champion,
        history: if cfg.record_history {
            Some(history)
        } else {
            None
        },
    }
}

mod tests {
    use crate::{Individual, Individuals};

    #[test]
    fn test_individuals() {
        // Initialize colletion of individuals
        let mut xs = Individuals::with_capacity(3, 10);
        for i in 0..10 {
            let i = i as f32;
            xs.push(Individual {
                values: Vec::from([i + 0.1, i + 0.2, i + 0.3]),
                fitness: i,
            });
            assert_eq!(xs.len(), i as u32 + 1);
        }
        assert_eq!(
            xs.get(0),
            Individual {
                values: Vec::from([0.1, 0.2, 0.3]),
                fitness: 0.0,
            }
        );
        assert_eq!(
            xs.get(1),
            Individual {
                values: Vec::from([1.1, 1.2, 1.3]),
                fitness: 1.0,
            }
        );
        assert_eq!(
            xs.get(9),
            Individual {
                values: Vec::from([9.1, 9.2, 9.3]),
                fitness: 9.0,
            }
        );
        xs.set(
            9,
            Individual {
                values: Vec::from([42.0, 69.0, 96.0]),
                fitness: 42.0,
            },
        );
        assert_eq!(
            xs.get(9),
            Individual {
                values: Vec::from([42.0, 69.0, 96.0]),
                fitness: 42.0,
            }
        );
    }
}
