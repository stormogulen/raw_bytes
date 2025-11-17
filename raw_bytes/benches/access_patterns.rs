// benches/access_patterns.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use raw_bytes::Container;
use bytemuck_derive::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Point3D {
    x: f64,
    y: f64,
    z: f64,
}

fn create_container(size: usize) -> Container<Point3D> {
    let mut c = Container::with_capacity(size);
    for i in 0..size {
        c.push(Point3D {
            x: i as f64,
            y: (i * 2) as f64,
            z: (i * 3) as f64,
        }).unwrap();
    }
    c
}

fn bench_individual_get(c: &mut Criterion) {
    let sizes = vec![100, 1_000, 10_000];
    
    let mut group = c.benchmark_group("individual_get");
    for size in sizes {
        let container = create_container(size);
        
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let mut sum = 0.0;
                for i in 0..container.len() {
                    sum += black_box(container.get(i).unwrap().x);
                }
                sum
            });
        });
    }
    group.finish();
}

fn bench_iterator(c: &mut Criterion) {
    let sizes = vec![100, 1_000, 10_000];
    
    let mut group = c.benchmark_group("iterator");
    for size in sizes {
        let container = create_container(size);
        
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let sum: f64 = container.iter()
                    .map(|p| black_box(p.x))
                    .sum();
                sum
            });
        });
    }
    group.finish();
}

fn bench_slice_access(c: &mut Criterion) {
    let sizes = vec![100, 1_000, 10_000];
    
    let mut group = c.benchmark_group("slice_access");
    for size in sizes {
        let container = create_container(size);
        
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let sum: f64 = container.as_slice()
                    .iter()
                    .map(|p| black_box(p.x))
                    .sum();
                sum
            });
        });
    }
    group.finish();
}

fn bench_index_syntax(c: &mut Criterion) {
    let sizes = vec![100, 1_000, 10_000];
    
    let mut group = c.benchmark_group("index_syntax");
    for size in sizes {
        let container = create_container(size);
        
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let mut sum = 0.0;
                for i in 0..container.len() {
                    sum += black_box(container[i].x);
                }
                sum
            });
        });
    }
    group.finish();
}

fn bench_write_operations(c: &mut Criterion) {
    let sizes = vec![100, 1_000, 10_000];
    
    let mut group = c.benchmark_group("write_operations");
    for size in sizes {
        group.bench_with_input(BenchmarkId::new("write_method", size), &size, |b, &s| {
            let mut container = create_container(s);
            b.iter(|| {
                for i in 0..container.len() {
                    container.write(i, Point3D { x: 1.0, y: 2.0, z: 3.0 }).unwrap();
                }
            });
        });
        
        group.bench_with_input(BenchmarkId::new("get_mut", size), &size, |b, &s| {
            let mut container = create_container(s);
            b.iter(|| {
                for i in 0..container.len() {
                    let p = container.get_mut(i).unwrap();
                    p.x = 1.0;
                    p.y = 2.0;
                    p.z = 3.0;
                }
            });
        });
        
        group.bench_with_input(BenchmarkId::new("mut_slice", size), &size, |b, &s| {
            let mut container = create_container(s);
            b.iter(|| {
                let slice = container.as_mut_slice().unwrap();
                for p in slice.iter_mut() {
                    p.x = 1.0;
                    p.y = 2.0;
                    p.z = 3.0;
                }
            });
        });
    }
    group.finish();
}

fn bench_push_operations(c: &mut Criterion) {
    let sizes = vec![100, 1_000, 10_000];
    
    let mut group = c.benchmark_group("push_operations");
    for size in sizes {
        group.bench_with_input(BenchmarkId::new("without_capacity", size), &size, |b, &s| {
            b.iter(|| {
                let mut container = Container::<Point3D>::new();
                for i in 0..s {
                    container.push(Point3D {
                        x: i as f64,
                        y: (i * 2) as f64,
                        z: (i * 3) as f64,
                    }).unwrap();
                }
            });
        });
        
        group.bench_with_input(BenchmarkId::new("with_capacity", size), &size, |b, &s| {
            b.iter(|| {
                let mut container = Container::<Point3D>::with_capacity(s);
                for i in 0..s {
                    container.push(Point3D {
                        x: i as f64,
                        y: (i * 2) as f64,
                        z: (i * 3) as f64,
                    }).unwrap();
                }
            });
        });
        
        group.bench_with_input(BenchmarkId::new("from_slice", size), &size, |b, &s| {
            let data: Vec<Point3D> = (0..s).map(|i| Point3D {
                x: i as f64,
                y: (i * 2) as f64,
                z: (i * 3) as f64,
            }).collect();
            
            b.iter(|| {
                Container::from_slice(&data)
            });
        });
    }
    group.finish();
}

#[cfg(feature = "mmap")]
fn bench_mmap_operations(c: &mut Criterion) {
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    let sizes = vec![100, 1_000, 10_000];
    
    let mut group = c.benchmark_group("mmap_operations");
    for size in sizes {
        // Create test file
        let data: Vec<Point3D> = (0..size).map(|i| Point3D {
            x: i as f64,
            y: (i * 2) as f64,
            z: (i * 3) as f64,
        }).collect();
        
        let mut file = NamedTempFile::new().unwrap();
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        file.write_all(bytes).unwrap();
        file.flush().unwrap();
        
        // Benchmark readonly
        group.bench_with_input(BenchmarkId::new("readonly_iter", size), &size, |b, _| {
            let container = Container::<Point3D>::mmap_readonly(file.path()).unwrap();
            b.iter(|| {
                let sum: f64 = container.iter().map(|p| black_box(p.x)).sum();
                sum
            });
        });
        
        // Benchmark readwrite
        group.bench_with_input(BenchmarkId::new("readwrite_read", size), &size, |b, _| {
            let container = Container::<Point3D>::mmap_readwrite(file.path()).unwrap();
            b.iter(|| {
                let sum: f64 = container.iter().map(|p| black_box(p.x)).sum();
                sum
            });
        });
        
        group.bench_with_input(BenchmarkId::new("readwrite_write", size), &size, |b, _| {
            let mut container = Container::<Point3D>::mmap_readwrite(file.path()).unwrap();
            b.iter(|| {
                let slice = container.as_mut_slice().unwrap();
                for p in slice.iter_mut() {
                    p.x = 999.0;
                }
            });
        });
    }
    group.finish();
}

criterion_group!(
    read_benches,
    bench_individual_get,
    bench_iterator,
    bench_slice_access,
    bench_index_syntax
);

criterion_group!(
    write_benches,
    bench_write_operations,
    bench_push_operations
);

#[cfg(feature = "mmap")]
criterion_group!(
    mmap_benches,
    bench_mmap_operations
);

// Register benchmark groups
#[cfg(feature = "mmap")]
criterion_main!(read_benches, write_benches, mmap_benches);

#[cfg(not(feature = "mmap"))]
criterion_main!(read_benches, write_benches);