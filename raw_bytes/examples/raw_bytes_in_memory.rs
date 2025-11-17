use raw_bytes::{Container, ContainerError};
use bytemuck_derive::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct Packet {
    id: u32,
    value: f32,
}

fn main() -> Result<(), ContainerError> {
    println!("=== In-Memory Example ===");

    let mut c = Container::<Packet>::new();

    c.push(Packet { id: 1, value: 10.0 })?;
    c.push(Packet { id: 2, value: 20.0 })?;
    c.push(Packet { id: 3, value: 30.0 })?;

    println!("len = {}", c.len());
    println!("first: {:?}", c.get(0)?);

    for i in 0..c.len() {
        println!("  [{}] {:?}", i, c.get(i)?);
    }

    Ok(())
}
