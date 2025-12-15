
//! Builder patterns for creating packed data structures

use std::marker::PhantomData;

/// Builder for creating packed data arrays with a fluent API
pub struct PackedDataBuilder<T> {
    data: Vec<T>,
    _phantom: PhantomData<T>,
}

impl<T> PackedDataBuilder<T> {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            _phantom: PhantomData,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            _phantom: PhantomData,
        }
    }

    pub fn push(mut self, item: T) -> Self {
        self.data.push(item);
        self
    }

    pub fn extend<I: IntoIterator<Item = T>>(mut self, items: I) -> Self {
        self.data.extend(items);
        self
    }

    pub fn build(self) -> Vec<T> {
        self.data
    }
}

impl<T> Default for PackedDataBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper for building entities with fixed-point coordinates
pub struct EntityBuilder<T> {
    items: Vec<T>,
}

impl<T> EntityBuilder<T> {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
        }
    }

    pub fn add(mut self, item: T) -> Self {
        self.items.push(item);
        self
    }

    pub fn build(self) -> Vec<T> {
        self.items
    }
}

impl<T> Default for EntityBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}
