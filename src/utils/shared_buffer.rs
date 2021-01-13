/*
 * opencl stream executor
 * Copyright (C) 2021 trivernis
 * See LICENSE for more information
 */
use ocl::{Buffer, OclPrm};
use parking_lot::Mutex;
use std::sync::Arc;

#[derive(Clone)]
pub struct SharedBuffer<T>
where
    T: OclPrm,
{
    inner: Arc<Mutex<Buffer<T>>>,
}

impl<T> SharedBuffer<T>
where
    T: OclPrm,
{
    /// Creates a new shared buffer with an inner ocl buffer
    pub fn new(buf: Buffer<T>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(buf)),
        }
    }

    /// Writes into the buffer
    pub fn write(&self, src: &[T]) -> ocl::Result<()> {
        let buffer = self.inner.lock();
        buffer.write(src).enq()
    }

    /// Reads from the buffer
    pub fn read(&self, dst: &mut [T]) -> ocl::Result<()> {
        let buffer = self.inner.lock();
        buffer.read(dst).enq()
    }

    /// Returns the inner buffer
    pub fn inner(&self) -> Arc<Mutex<Buffer<T>>> {
        Arc::clone(&self.inner)
    }
}
