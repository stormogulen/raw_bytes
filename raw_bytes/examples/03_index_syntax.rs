//! Using index syntax for convenient access

use raw_bytes::Container;
use bytemuck_derive::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct Vertex {
    x: f32,
    y: f32,
    z: f32,
}

fn main() {
    println!("=== Index Syntax Example ===\n");

    let data = vec![
        Vertex { x: 1.0, y: 0.0, z: 0.0 },
        Vertex { x: 0.0, y: 1.0, z: 0.0 },
        Vertex { x: 0.0, y: 0.0, z: 1.0 },
    ];
    
    let mut c = Container::from_slice(&data);

    // Read using index syntax (implements Index trait)
    println!("Vertex 0: {:?}", c[0]);
    println!("Vertex 1: {:?}", c[1]);
    println!("Vertex 2: {:?}", c[2]);

    // Modify using index syntax (implements IndexMut trait)
    println!("\nModifying vertex 1...");
    c[1].y = 2.0;
    c[1].z = 0.5;
    
    println!("Vertex 1 after: {:?}", c[1]);

    // Use in calculations
    let sum_x: f32 = (0..c.len()).map(|i| c[i].x).sum();
    let sum_y: f32 = (0..c.len()).map(|i| c[i].y).sum();
    let sum_z: f32 = (0..c.len()).map(|i| c[i].z).sum();
    
    println!("\nSum of all components:");
    println!("  x: {}, y: {}, z: {}", sum_x, sum_y, sum_z);
}
