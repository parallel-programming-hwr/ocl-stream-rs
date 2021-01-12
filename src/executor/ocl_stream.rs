/*
 * opencl stream executor
 * Copyright (C) 2021 trivernis
 * See LICENSE for more information
 */

use crossbeam_channel::{Receiver, Sender};

use crate::utils::result::{OCLStreamError, OCLStreamResult};

/// Creates a new OCLStream with the corresponding sender
/// to communicate between the scheduler thread and the receiver thread
pub fn create<T>() -> (OCLStream<T>, OCLStreamSender<T>)
where
    T: Send + Sync,
{
    let (tx, rx) = crossbeam_channel::unbounded();
    let stream = OCLStream { rx };
    let sender = OCLStreamSender { tx };

    (stream, sender)
}

/// Receiver for OCL Data
#[derive(Clone, Debug)]
pub struct OCLStream<T>
where
    T: Send + Sync,
{
    rx: Receiver<OCLStreamResult<T>>,
}

impl<T> OCLStream<T>
where
    T: Send + Sync,
{
    /// Reads the next value from the channel
    pub fn next(&mut self) -> Result<T, OCLStreamError> {
        self.rx.recv()?
    }

    /// Returns if there is a value in the channel
    pub fn has_next(&self) -> bool {
        !self.rx.is_empty()
    }
}

/// Sender for OCL Data
pub struct OCLStreamSender<T>
where
    T: Send + Sync,
{
    tx: Sender<OCLStreamResult<T>>,
}

impl<T> Clone for OCLStreamSender<T>
where
    T: Send + Sync,
{
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
        }
    }
}

impl<T> OCLStreamSender<T>
where
    T: Send + Sync,
{
    /// Sends a value into the channel
    pub fn send(&self, value: T) -> OCLStreamResult<()> {
        self.tx
            .send(Ok(value))
            .map_err(|_| OCLStreamError::SendError)
    }

    /// Sends an error into the channel
    pub fn err(&self, err: OCLStreamError) -> OCLStreamResult<()> {
        self.tx
            .send(Err(err))
            .map_err(|_| OCLStreamError::SendError)
    }
}
