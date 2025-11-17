//use raw_bytes::Container;
//use bytemuck::Pod;
use bytemuck_derive::Zeroable;
use bytemuck_derive::Pod;
#[cfg(feature = "std")]
use std::{fs::File, io::Write, path::Path};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct Packet {
    id: u32,
    value: f32,
}

#[cfg(feature = "std")]
fn ensure_file(path: &Path, count: usize) -> std::io::Result<()> {
    if !path.exists() {
        let mut file = File::create(path)?;
        let zero_bytes = vec![0u8; count * std::mem::size_of::<Packet>()];
        file.write_all(&zero_bytes)?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Mmap Read-Only Example ===");

    #[cfg(feature = "mmap")]
    {
        let path = Path::new("data_readonly.bin");
        ensure_file(path, 10)?; // create file with space for 10 Packet elements

        let c = raw_bytes::Container::<Packet>::mmap_readonly(path)?;
        println!("len = {}", c.len());
        println!("first: {:?}", c.get(0)?);
    }

    Ok(())
}
