//! Basic in-memory container usage

use bytemuck_derive::{Pod, Zeroable};
use raw_bytes::{Container, ContainerError};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct Packet {
    id: u32,
    value: f32,
}

fn main() -> Result<(), ContainerError> {
    println!("=== Basic In-Memory Example ===\n");

    // Create empty container
    let mut c = Container::<Packet>::new();

    // Push some packets
    c.push(Packet { id: 1, value: 10.0 })?;
    c.push(Packet { id: 2, value: 20.0 })?;
    c.push(Packet { id: 3, value: 30.0 })?;

    println!("Container has {} packets", c.len());

    // Read individual packets
    println!("\nIterating by index:");
    for i in 0..c.len() {
        println!("  [{}] {:?}", i, c.get(i)?);
    }

    // Modify a packet
    println!("\nModifying packet at index 1...");
    c.write(
        1,
        Packet {
            id: 99,
            value: 999.0,
        },
    )?;
    println!("  After: {:?}", c.get(1)?);

    // Use iterator
    println!("\nIterating with iterator:");
    for (i, packet) in c.iter().enumerate() {
        println!("  [{}] id={}, value={}", i, packet.id, packet.value);
    }

    // Calculate sum using iterator
    let total: f32 = c.iter().map(|p| p.value).sum();
    println!("\nTotal value: {}", total);

    Ok(())
}
