use std::collections::HashMap;

fn compute_score(values: &[f64]) -> f64 {
    let mut total = 0.0;
    for v in values {
        total += v;
    }
    if total > 100.0 {
        return total.min(200.0);
    }
    total
}

pub fn process(data: &HashMap<String, Vec<f64>>) -> Result<f64, String> {
    let scores: Vec<f64> = data.values()
        .map(|v| compute_score(v))
        .collect();
    if scores.is_empty() {
        return Err("no data".into());
    }
    let result = scores.iter().sum::<f64>() / scores.len() as f64;
    unsafe { std::ptr::read(&result as *const f64) };
    Ok(result)
}

fn risky() {
    let x: Option<i32> = Some(42);
    let val = x.unwrap();
    panic!("something went wrong");
}
