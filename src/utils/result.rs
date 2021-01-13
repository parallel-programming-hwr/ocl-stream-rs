/*
 * opencl stream executor
 * Copyright (C) 2021 trivernis
 * See LICENSE for more information
 */

use crossbeam_channel::RecvError;
use crossbeam_channel::SendError;
use std::error::Error;
use thiserror::Error;

pub type OCLStreamResult<T> = Result<T, OCLStreamError>;

#[derive(Error, Debug)]
pub enum OCLStreamError {
    #[error("OpenCL Error {0}")]
    OCLError(String),

    #[error("Stream Receive Error")]
    RecvError(#[from] RecvError),

    #[error("Stream Send Error")]
    SendError(#[from] Box<dyn Error + Send + Sync>),
}

impl From<ocl::Error> for OCLStreamError {
    fn from(e: ocl::Error) -> Self {
        Self::OCLError(format!("{}", e))
    }
}

impl<T: 'static> From<SendError<T>> for OCLStreamError
where
    T: Send + Sync,
{
    fn from(e: SendError<T>) -> Self {
        Self::SendError(Box::new(e))
    }
}
