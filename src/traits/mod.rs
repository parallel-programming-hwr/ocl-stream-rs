/*
 * opencl stream executor
 * Copyright (C) 2021 trivernis
 * See LICENSE for more information
 */

pub mod to_ocl_buffer;
pub mod to_shared_buffer;
pub use to_ocl_buffer::*;
pub use to_shared_buffer::*;
