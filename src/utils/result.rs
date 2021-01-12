use crossbeam_channel::RecvError;
use std::error::Error;
use std::fmt::{self, Display, Formatter};

pub type OCLStreamResult<T> = Result<T, OCLStreamError>;

#[derive(Debug)]
pub enum OCLStreamError {
    OCLError(ocl::Error),
    RecvError(RecvError),
    SendError,
}

impl Display for OCLStreamError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            OCLStreamError::OCLError(e) => write!(f, "OCL Error: {}", e),
            OCLStreamError::RecvError(e) => write!(f, "Stream Receive Error: {}", e),
            OCLStreamError::SendError => write!(f, "Stream Send Error"),
        }
    }
}

impl Error for OCLStreamError {}

impl From<ocl::Error> for OCLStreamError {
    fn from(e: ocl::Error) -> Self {
        Self::OCLError(e)
    }
}

impl From<RecvError> for OCLStreamError {
    fn from(e: RecvError) -> Self {
        Self::RecvError(e)
    }
}
