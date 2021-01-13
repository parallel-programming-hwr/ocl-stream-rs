/*
 * opencl stream executor
 * Copyright (C) 2021 trivernis
 * See LICENSE for more information
 */

use crate::executor::context::ExecutorContext;
use crate::executor::stream::{OCLStream, OCLStreamSender};
use crate::utils::result::OCLStreamResult;
use crate::utils::shared_buffer::SharedBuffer;
use ocl::{OclPrm, ProQue};
use std::sync::Arc;
use std::thread;

pub mod context;
pub mod stream;

/// Stream executor for OpenCL Programs
#[derive(Clone)]
pub struct OCLStreamExecutor {
    pro_que: ProQue,
    concurrency: usize,
}

impl OCLStreamExecutor {
    /// Creates a new OpenCL Stream executor
    /// ```rust
    /// use ocl::ProQue;
    /// use ocl_stream::OCLStreamExecutor;
    /// let pro_que = ProQue::builder().src("__kernel void bench_int() {}").build().unwrap();
    /// let executor = OCLStreamExecutor::new(pro_que);
    /// ```
    pub fn new(pro_que: ProQue) -> Self {
        Self {
            pro_que,
            concurrency: 1,
        }
    }

    /// Sets how many threads should be used to schedule kernels on
    /// the gpu. Using multiple threads reduces the idle time of the gpu.
    /// While one kernel is running, the next one can be prepared in a
    /// different thread. A value of 0 means that the number of cpu cores should be used.
    pub fn set_concurrency(&mut self, mut num_tasks: usize) {
        if num_tasks == 0 {
            num_tasks = num_cpus::get();
        }
        self.concurrency = num_tasks;
    }

    /// Executes a closure in the ocl context with a bounded channel
    pub fn execute_bounded<F, T>(&self, size: usize, func: F) -> OCLStream<T>
    where
        F: Fn(ExecutorContext<T>) -> OCLStreamResult<()> + Send + Sync + 'static,
        T: Send + Sync + 'static,
    {
        let (stream, sender) = stream::bounded(size);
        self.execute(func, sender);

        stream
    }

    /// Returns the inner pro_que object
    pub fn pro_que(&self) -> &ProQue {
        &self.pro_que
    }

    /// Executes a closure in the ocl context with an unbounded channel
    /// for streaming
    pub fn execute_unbounded<F, T>(&self, func: F) -> OCLStream<T>
    where
        F: Fn(ExecutorContext<T>) -> OCLStreamResult<()> + Send + Sync + 'static,
        T: Send + Sync + 'static,
    {
        let (stream, sender) = stream::unbounded();
        self.execute(func, sender);

        stream
    }

    /// Executes a closure in the ocl context
    fn execute<F, T>(&self, func: F, sender: OCLStreamSender<T>)
    where
        F: Fn(ExecutorContext<T>) -> OCLStreamResult<()> + Send + Sync + 'static,
        T: Send + Sync + 'static,
    {
        let func = Arc::new(func);

        for task_id in 0..(self.concurrency) {
            let func = Arc::clone(&func);
            let context = self.build_context(task_id, sender.clone());

            thread::Builder::new()
                .name(format!("ocl-{}", task_id))
                .spawn(move || {
                    let sender = context.sender().clone();

                    if let Err(e) = func(context) {
                        if let Err(e) = sender.err(e) {
                            panic!("Failed to forward error to receiver: {}", e);
                        }
                    }
                })
                .expect("Failed to spawn ocl thread");
        }
    }

    /// Creates a new shard buffer with a given length
    pub fn create_shared_buffer<T>(&self, len: usize) -> ocl::Result<SharedBuffer<T>>
    where
        T: OclPrm,
    {
        let buffer = self.pro_que.buffer_builder().len(len).build()?;
        Ok(SharedBuffer::new(buffer))
    }

    /// Builds the executor context for the executor
    fn build_context<T>(&self, task_id: usize, sender: OCLStreamSender<T>) -> ExecutorContext<T>
    where
        T: Send + Sync,
    {
        ExecutorContext::new(self.pro_que.clone(), task_id, sender)
    }
}
