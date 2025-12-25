//! Memory-mapped read-only file example

use bytemuck_derive::{Pod, Zeroable};
use raw_bytes::Container;

#[cfg(all(feature = "std", feature = "mmap"))]
use std::{fs::File, io::Write, path::Path};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
struct Record {
    timestamp: u64,
    sensor_id: u32,
    value: f32,
}

#[cfg(all(feature = "std", feature = "mmap"))]
fn create_sample_data(path: &Path) -> std::io::Result<()> {
    let records = vec![
        Record {
            timestamp: 1000,
            sensor_id: 1,
            value: 23.5,
        },
        Record {
            timestamp: 2000,
            sensor_id: 1,
            value: 24.1,
        },
        Record {
            timestamp: 3000,
            sensor_id: 2,
            value: 18.7,
        },
        Record {
            timestamp: 4000,
            sensor_id: 2,
            value: 19.2,
        },
        Record {
            timestamp: 5000,
            sensor_id: 3,
            value: 21.0,
        },
    ];

    let mut file = File::create(path)?;
    let bytes: &[u8] = bytemuck::cast_slice(&records);
    file.write_all(bytes)?;
    file.flush()?;

    println!("Created sample file with {} records", records.len());
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Memory-Mapped Read-Only Example ===\n");

    #[cfg(all(feature = "std", feature = "mmap"))]
    {
        let path = Path::new("sensor_data_readonly.bin");

        // Create sample data file
        create_sample_data(path)?;

        // Open as memory-mapped read-only
        let c = Container::<Record>::mmap_readonly(path)?;
        println!("\nOpened mmap file with {} records", c.len());

        // Read data efficiently
        println!("\nAll records:");
        for (i, record) in c.iter().enumerate() {
            println!(
                "  [{}] ts={}, sensor={}, value={:.1}",
                i, record.timestamp, record.sensor_id, record.value
            );
        }

        // Filter by sensor using slice
        let sensor_id = 2;
        println!("\nRecords for sensor {}:", sensor_id);
        for record in c.as_slice().iter().filter(|r| r.sensor_id == sensor_id) {
            println!("  ts={}, value={:.1}", record.timestamp, record.value);
        }

        // Calculate average
        let avg: f32 = c.iter().map(|r| r.value).sum::<f32>() / c.len() as f32;
        println!("\nAverage value: {:.2}", avg);

        // Attempting to modify will fail (commented out to avoid panic)
        // c.write(0, Record { timestamp: 9999, sensor_id: 99, value: 99.9 })?; // ERROR!

        println!("\nRead-only access works correctly");

        // Cleanup
        std::fs::remove_file(path)?;
    }

    #[cfg(not(all(feature = "std", feature = "mmap")))]
    println!("This example requires 'std' and 'mmap' features");

    Ok(())
}
