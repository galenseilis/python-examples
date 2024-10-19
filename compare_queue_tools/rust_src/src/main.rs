use rand::Rng;

#[derive(Debug, PartialEq, PartialOrd)]
struct Server {
    end_time: f64,
}

impl Eq for Server {}

impl Ord for Server {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

fn run_trial(_seed: usize, arrival_rate: f64, service_rate: f64, num_servers: usize, max_sim_time: f64, warmup: f64) -> f64 {
    let mut outcome = 0.0;
    let mut rng = rand::thread_rng();
    let mut servers_end: std::collections::BinaryHeap<Server> = std::collections::BinaryHeap::new();
    let mut records: Vec<Vec<f64>> = Vec::new();

    for _ in 0..num_servers {
        servers_end.push(Server { end_time: 0.0 });
    }

    let mut arrival_date = 0.0;
    let mut sum_waits = 0.0;
    let mut waits: Vec<f64> = Vec::new();

    while arrival_date < max_sim_time {
        let r1: f64 = rng.gen();
        let r2: f64 = rng.gen();

        let log_r1 = (-r1.ln()) / arrival_rate;
        let log_r2 = (-r2.ln()) / service_rate;

        arrival_date += log_r1;

        let service_time = log_r2;
        let service_start_date = arrival_date.max(servers_end.peek().unwrap().end_time);
        let service_end_date = service_start_date + service_time;
        let wait = service_start_date - arrival_date;

        servers_end.pop();
        servers_end.push(Server {
            end_time: service_end_date,
        });

        records.push(vec![arrival_date, wait]);
    }

    for record in records.iter().filter(|record| record[0] > warmup) {
        waits.push(record[1]);
    }

    for wait in waits.iter() {
        sum_waits += wait;
    }

    if !waits.is_empty() {
        outcome = sum_waits / waits.len() as f64;
    }

    outcome
}

fn main() {
    let num_servers = 3;
    let num_trials = 20;
    let arrival_rate = 10.0;
    let service_rate = 4.0;
    let max_sim_time = 800.0;
    let warmup = 100.0;

    let mean_waits: Vec<f64> = (0..num_trials)
        .map(|seed| run_trial(seed, arrival_rate, service_rate, num_servers, max_sim_time, warmup))
        .collect();

    let sum_mean_waits: f64 = mean_waits.iter().sum();
    let solution = sum_mean_waits / mean_waits.len() as f64;

    // println!("Solution: {}", solution);
}

