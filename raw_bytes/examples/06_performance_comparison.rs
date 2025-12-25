//! Compare different access patterns

use bytemuck_derive::{Pod, Zeroable};
use raw_bytes::Container;
use std::time::Instant;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DataPoint {
    x: f64,
    y: f64,
    z: f64,
}

fn main() {
    println!("=== Performance Comparison ===\n");

    const N: usize = 1_000_000;

    // Create test data
    println!("Creating {} data points...", N);
    let mut c = Container::<DataPoint>::with_capacity(N);
    for i in 0..N {
        c.push(DataPoint {
            x: i as f64,
            y: (i * 2) as f64,
            z: (i * 3) as f64,
        })
        .unwrap();
    }

    // Method 1: Individual get() calls
    let start = Instant::now();
    let mut sum1 = 0.0;
    for i in 0..c.len() {
        sum1 += c.get(i).unwrap().x;
    }
    let time1 = start.elapsed();
    println!("Individual get() calls: {:?}", time1);
    println!("  Sum: {}", sum1);

    // Method 2: Using iterator
    let start = Instant::now();
    let sum2: f64 = c.iter().map(|p| p.x).sum();
    let time2 = start.elapsed();
    println!("\nUsing iterator: {:?}", time2);
    println!("  Sum: {}", sum2);

    // Method 3: Direct slice access
    let start = Instant::now();
    let sum3: f64 = c.as_slice().iter().map(|p| p.x).sum();
    let time3 = start.elapsed();
    println!("\nDirect slice access: {:?}", time3);
    println!("  Sum: {}", sum3);

    println!("\nPerformance gains:");
    println!(
        "  Iterator vs get(): {:.2}x faster",
        time1.as_secs_f64() / time2.as_secs_f64()
    );
    println!(
        "  Slice vs get(): {:.2}x faster",
        time1.as_secs_f64() / time3.as_secs_f64()
    );
}
