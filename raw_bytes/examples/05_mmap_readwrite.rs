//! Memory-mapped read-write file example

use bytemuck_derive::{Pod, Zeroable};
use raw_bytes::Container;

#[cfg(all(feature = "std", feature = "mmap"))]
use std::{fs::File, io::Write, path::Path};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct GameScore {
    player_id: u32,
    score: u32,
    level: u16,
    flags: u16,
}

#[cfg(all(feature = "std", feature = "mmap"))]
fn create_initial_scores(path: &Path) -> std::io::Result<()> {
    let scores = vec![
        GameScore {
            player_id: 1,
            score: 1000,
            level: 5,
            flags: 0,
        },
        GameScore {
            player_id: 2,
            score: 2500,
            level: 8,
            flags: 1,
        },
        GameScore {
            player_id: 3,
            score: 500,
            level: 2,
            flags: 0,
        },
        GameScore {
            player_id: 4,
            score: 3200,
            level: 10,
            flags: 3,
        },
    ];

    let mut file = File::create(path)?;
    let bytes: &[u8] = bytemuck::cast_slice(&scores);
    file.write_all(bytes)?;
    file.flush()?;

    println!("Created initial scores file with {} players", scores.len());
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Memory-Mapped Read-Write Example ===\n");

    #[cfg(all(feature = "std", feature = "mmap"))]
    {
        let path = Path::new("game_scores.bin");

        // Create initial data
        create_initial_scores(path)?;

        // Open as memory-mapped read-write
        let mut c = Container::<GameScore>::mmap_readwrite(path)?;
        println!("\nOpened mmap file with {} player scores", c.len());

        // Display initial state
        println!("\nInitial scores:");
        for (_i, score) in c.iter().enumerate() {
            println!(
                "  Player {}: score={}, level={}, flags={}",
                score.player_id, score.score, score.level, score.flags
            );
        }

        // Update a score (changes persist to file!)
        println!("\nUpdating player 2's score...");
        c.write(
            1,
            GameScore {
                player_id: 2,
                score: 5000, // increased!
                level: 9,    // leveled up!
                flags: 1,
            },
        )?;

        // Batch update using mutable slice
        println!("Adding bonus points to all players...");
        let mut_slice = c.as_mut_slice()?;
        for score in mut_slice.iter_mut() {
            score.score += 100; // bonus!
        }

        // Display final state
        println!("\nFinal scores:");
        for (_i, score) in c.iter().enumerate() {
            println!(
                "  Player {}: score={}, level={}, flags={}",
                score.player_id, score.score, score.level, score.flags
            );
        }

        // Find top scorer
        if let Some(top) = c.as_slice().iter().max_by_key(|s| s.score) {
            println!(
                "\n🏆 Top scorer: Player {} with {} points!",
                top.player_id, top.score
            );
        }

        println!("\n✓ Changes persisted to file!");

        // Note: Cannot push to mmap storage
        // c.push(GameScore { player_id: 5, score: 0, level: 1, flags: 0 })?; // ERROR!

        // Cleanup
        std::fs::remove_file(path)?;
    }

    #[cfg(not(all(feature = "std", feature = "mmap")))]
    println!("This example requires 'std' and 'mmap' features");

    Ok(())
}
