/*
 * opencl stream executor
 * Copyright (C) 2021 trivernis
 * See LICENSE for more information
 */

use crate::executor::stream::OCLStreamSender;
use ocl::ProQue;

/// Context passed to the executing closure
/// to provide additional information and
/// access to the ProQue.
#[derive(Clone)]
pub struct ExecutorContext<T>
where
    T: Send + Sync,
{
    pro_que: ProQue,
    sender: OCLStreamSender<T>,
    task_id: usize,
}

impl<T> ExecutorContext<T>
where
    T: Send + Sync,
{
    /// Creates a new executor context.
    pub fn new(pro_que: ProQue, task_id: usize, sender: OCLStreamSender<T>) -> Self {
        Self {
            pro_que,
            task_id,
            sender,
        }
    }

    /// Returns the ProQue
    pub fn pro_que(&self) -> &ProQue {
        &self.pro_que
    }

    /// Returns the Sender
    pub fn sender(&self) -> &OCLStreamSender<T> {
        &self.sender
    }

    /// Returns the unique task id of the scheduled
    /// task
    pub fn task_id(&self) -> usize {
        self.task_id
    }
}
