//! Demonstrating bulk operations and capacity management

use raw_bytes::{Container, ContainerError};
use bytemuck_derive::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct Point {
    x: f32,
    y: f32,
}

fn main() -> Result<(), ContainerError> {
    println!("=== Bulk Operations Example ===\n");

    // Pre-allocate capacity for better performance
    let c = Container::<Point>::with_capacity(1000);
    println!("Created container with capacity for 1000 points {}", c.len());

    // Create from slice
    let initial_points = vec![
        Point { x: 0.0, y: 0.0 },
        Point { x: 1.0, y: 1.0 },
        Point { x: 2.0, y: 2.0 },
    ];
    let mut c2 = Container::from_slice(&initial_points);
    println!("\nCreated container from {} points", c2.len());

    // Extend with more data
    let more_points = vec![
        Point { x: 3.0, y: 3.0 },
        Point { x: 4.0, y: 4.0 },
    ];
    c2.extend_from_slice(&more_points)?;
    println!("Extended to {} points", c2.len());

    // Use slice access for efficient processing
    let slice = c2.as_slice();
    println!("\nProcessing via slice:");
    for (i, point) in slice.iter().enumerate() {
        println!("  Point {}: ({}, {})", i, point.x, point.y);
    }

    // Mutable slice for batch updates
    println!("\nScaling all points by 2.0...");
    let mut_slice = c2.as_mut_slice()?;
    for point in mut_slice.iter_mut() {
        point.x *= 2.0;
        point.y *= 2.0;
    }

    println!("\nAfter scaling:");
    for (i, point) in c2.iter().enumerate() {
        println!("  Point {}: ({}, {})", i, point.x, point.y);
    }

    // Clear the container
    c2.clear()?;
    println!("\nCleared container, now has {} points", c2.len());

    Ok(())
}