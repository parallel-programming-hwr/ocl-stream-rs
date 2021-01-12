/*
 * opencl stream executor
 * Copyright (C) 2021 trivernis
 * See LICENSE for more information
 */

use crossbeam_channel::RecvError;
use thiserror::Error;

pub type OCLStreamResult<T> = Result<T, OCLStreamError>;

#[derive(Error, Debug)]
pub enum OCLStreamError {
    #[error("OpenCL Error {0}")]
    OCLError(String),

    #[error("Stream Receive Error")]
    RecvError(#[from] RecvError),

    #[error("Stream Send Error")]
    SendError,
}

impl From<ocl::Error> for OCLStreamError {
    fn from(e: ocl::Error) -> Self {
        Self::OCLError(format!("{}", e))
    }
}
