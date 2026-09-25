use std::fs;

struct Model {
    n_in: usize,
    n_hidden: usize,
    params: Vec<f32>,
}

impl Model {
    fn w1(&self) -> &[f32] {
        &self.params[..self.n_hidden * self.n_in]
    }
    fn b1(&self) -> &[f32] {
        let start = self.n_hidden * self.n_in;
        &self.params[start..start + self.n_hidden]
    }
    fn w2(&self) -> &[f32] {
        let start = self.n_hidden * self.n_in + self.n_hidden;
        &self.params[start..start + self.n_hidden]
    }
    fn b2(&self) -> f32 {
        self.params[self.params.len() - 1]
    }
}

fn bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_u32(bytes: &[u8], offset: usize) -> usize {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize
}

fn load_model(path: &str) -> Model {
    let bytes = fs::read(path).expect("could not read weights file");
    let n_in = read_u32(&bytes, 0);
    let n_hidden = read_u32(&bytes, 4);
    assert_eq!(read_u32(&bytes, 8), 1, "this engine only supports 1 output");

    let params = bytes_to_f32s(&bytes[12..]);
    assert_eq!(params.len(), n_hidden * n_in + 2 * n_hidden + 1, "weights file is the wrong size");
    Model { n_in, n_hidden, params }
}

/// Load a file of scans
fn load_cases(path: &str, per_case: usize) -> (Vec<f32>, usize) {
    let bytes = fs::read(path).expect("could not read scans file");
    let count = read_u32(&bytes, 0);
    let nums = bytes_to_f32s(&bytes[4..]);
    assert_eq!(nums.len(), count * per_case, "scans file is the wrong size");
    (nums, count)
}

fn write_f32s(path: &str, data: &[f32]) {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, bytes).expect("could not write file");
}

/// Each hidden neuron's value before ReLU
fn hidden_sums(m: &Model, x: &[f32]) -> Vec<f32> {
    let (w1, b1) = (m.w1(), m.b1());
    (0..m.n_hidden)
        .map(|i| {
            let row = &w1[i * m.n_in..(i + 1) * m.n_in];
            let mut sum = b1[i];
            for j in 0..m.n_in {
                sum += row[j] * x[j];
            }
            sum
        })
        .collect()
}

/// Forward pass
fn forward(m: &Model, x: &[f32]) -> f32 {
    let z = hidden_sums(m, x);
    let w2 = m.w2();
    let mut score = m.b2();
    for i in 0..m.n_hidden {
        score += w2[i] * z[i].max(0.0); // ReLU, then output layer
    }
    score
}

fn gradient(m: &Model, x: &[f32]) -> Vec<f32> {
    let z = hidden_sums(m, x);
    let (w1, w2) = (m.w1(), m.w2());
    let mut grad = vec![0.0f32; m.n_in];
    for i in 0..m.n_hidden {
        if z[i] > 0.0 {
            let row = &w1[i * m.n_in..(i + 1) * m.n_in];
            for j in 0..m.n_in {
                grad[j] += w2[i] * row[j];
            }
        }
    }
    grad
}

fn fgsm(x: &[f32], grad: &[f32], eps: f32, push_down: bool) -> Vec<f32> {
    let dir = if push_down { -1.0 } else { 1.0 };
    x.iter()
        .zip(grad)
        .map(|(&xi, &gi)| (xi + dir * eps * gi.signum()).clamp(0.0, 1.0))
        .collect()
}

// experiments 
fn main() {
    let model = load_model("../weights.bin");
    let n = model.n_in;
    println!("Model: {} inputs -> {} hidden -> 1 output", n, model.n_hidden);

    let (cases, count) = load_cases("../test_cases.bin", n + 2);
    let mut max_diff = 0.0f32;
    for k in 0..count {
        let c = &cases[k * (n + 2)..(k + 1) * (n + 2)];
        max_diff = max_diff.max((forward(&model, &c[..n]) - c[n + 1]).abs());
    }
    println!("Parity with Python on {count} scans: largest difference {max_diff:.1e}");

    let (all, total) = load_cases("../test_all.bin", n + 1);
    let levels = [1.0f32, 2.0, 4.0, 8.0, 16.0]; 
    let mut flips = vec![[0usize; 2]; levels.len()]; 
    let mut correct = [0usize; 2];
    let mut times_on = vec![0usize; model.n_hidden];
    let mut pred_err_sum = 0.0f32;
    let mut best: Option<(usize, usize)> = None;

    for k in 0..total {
        let s = &all[k * (n + 1)..(k + 1) * (n + 1)];
        let x = &s[..n];
        let label = s[n] as usize;

        for (i, z) in hidden_sums(&model, x).iter().enumerate() {
            if *z > 0.0 {
                times_on[i] += 1;
            }
        }

        let score = forward(&model, x);
        if (score > 0.0) != (label == 1) {
            continue; 
        }
        correct[label] += 1;

        let grad = gradient(&model, x);
        let push_down = label == 1;
        let dir = if push_down { -1.0 } else { 1.0 };
        let l1: f32 = grad.iter().map(|g| g.abs()).sum();

        let mut first_flip = None;
        for (e, &lvl) in levels.iter().enumerate() {
            let adv = fgsm(x, &grad, lvl / 255.0, push_down);
            let new_score = forward(&model, &adv);
            if e == 0 {
                let predicted = score + dir * (lvl / 255.0) * l1;
                pred_err_sum += (predicted - new_score).abs();
            }
            if (new_score > 0.0) != (score > 0.0) {
                flips[e][label] += 1;
                first_flip.get_or_insert(e);
            }
        }
        if let (1, Some(e)) = (label, first_flip) {
            if best.map_or(true, |(_, b)| e < b) {
                best = Some((k, e));
            }
        }
    }

    let attacked = correct[0] + correct[1];
    println!("\nTest set: {total} scans. Correct (and attacked): {} normal, {} pneumonia", correct[0], correct[1]);
    println!("Linear prediction at eps=1/255: average error {:.4} in score", pred_err_sum / attacked as f32);

    println!("\n  eps     normal->pneumonia   pneumonia->normal   overall");
    let pct = |a: usize, b: usize| if b == 0 { 0.0 } else { 100.0 * a as f32 / b as f32 };
    let mut csv = String::from("eps_levels,normal_to_pneumonia,pneumonia_to_normal,overall\n");
    for (e, &lvl) in levels.iter().enumerate() {
        let (a, b) = (pct(flips[e][0], correct[0]), pct(flips[e][1], correct[1]));
        let o = pct(flips[e][0] + flips[e][1], attacked);
        println!("{lvl:>3}/255   {a:>14.1}%   {b:>16.1}%   {o:>6.1}%");
        csv += &format!("{lvl},{a},{b},{o}\n");
    }

    let dead = times_on.iter().filter(|&&t| t == 0).count();
    let always = times_on.iter().filter(|&&t| t == total).count();
    println!(
        "\nHidden neurons across all {total} scans: {dead} never on (dead), {always} always on, {} switch",
        model.n_hidden - dead - always
    );

    // plotting
    fs::create_dir_all("../results").expect("could not create results folder");
    fs::write("../results/flip_rates.csv", csv).expect("could not write csv");

    if let Some((k, e)) = best {
        let x = &all[k * (n + 1)..k * (n + 1) + n];
        let grad = gradient(&model, x);
        let adv = fgsm(x, &grad, levels[e] / 255.0, true);
        let mut out = Vec::with_capacity(3 * n + 3);
        out.extend_from_slice(x);
        out.extend_from_slice(&adv);
        out.extend_from_slice(&grad);
        out.extend_from_slice(&[forward(&model, x), forward(&model, &adv), levels[e]]);
        write_f32s("../results/example.bin", &out);
        println!("\nSaved example: test scan {k}, pneumonia -> normal at eps = {}/255", levels[e]);
    }
}