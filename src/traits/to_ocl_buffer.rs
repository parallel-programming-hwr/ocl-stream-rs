/*
 * opencl stream executor
 * Copyright (C) 2021 trivernis
 * See LICENSE for more information
 */

use ocl::{Buffer, OclPrm, ProQue};

pub trait ToOclBuffer<T>
where
    T: OclPrm,
{
    fn to_ocl_buffer(&self, pro_que: &ProQue) -> ocl::Result<Buffer<T>>;
}

impl<T> ToOclBuffer<T> for Vec<T>
where
    T: OclPrm,
{
    fn to_ocl_buffer(&self, pro_que: &ProQue) -> ocl::Result<Buffer<T>> {
        let buffer = pro_que.buffer_builder().len(self.len()).build()?;
        buffer.write(&self[..]).enq()?;

        Ok(buffer)
    }
}
