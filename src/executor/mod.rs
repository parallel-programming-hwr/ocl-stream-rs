/*
 * opencl stream executor
 * Copyright (C) 2021 trivernis
 * See LICENSE for more information
 */

use crate::executor::context::ExecutorContext;
use crate::executor::ocl_stream::{OCLStream, OCLStreamSender};
use crate::utils::result::OCLStreamResult;
use ocl::ProQue;
use scheduled_thread_pool::ScheduledThreadPool;
use std::sync::Arc;

pub mod context;
pub mod ocl_stream;

/// Stream executor for OpenCL Programs
#[derive(Clone)]
pub struct OCLStreamExecutor {
    pro_que: ProQue,
    pool: Arc<ScheduledThreadPool>,
    concurrency: usize,
}

impl OCLStreamExecutor {
    /// Creates a new OpenCL Stream executor
    pub fn new(pro_que: ProQue) -> Self {
        Self {
            pro_que,
            pool: Arc::new(ScheduledThreadPool::new(num_cpus::get())),
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

    /// Executes a closure in the ocl context
    pub fn execute<F, T>(&self, func: F) -> OCLStream<T>
    where
        F: Fn(ExecutorContext<T>) -> OCLStreamResult<()> + Send + Sync + 'static,
        T: Send + Sync + 'static,
    {
        let (stream, sender) = ocl_stream::create();
        let func = Arc::new(func);

        for task_id in 0..(self.concurrency) {
            let func = Arc::clone(&func);
            let context = self.build_context(task_id, sender.clone());

            self.pool.execute(move || {
                let sender2 = context.sender().clone();

                if let Err(e) = func(context) {
                    sender2.err(e).unwrap();
                }
            });
        }

        stream
    }

    /// Builds the executor context for the executor
    fn build_context<T>(&self, task_id: usize, sender: OCLStreamSender<T>) -> ExecutorContext<T>
    where
        T: Send + Sync,
    {
        ExecutorContext::new(self.pro_que.clone(), task_id, sender)
    }
}
