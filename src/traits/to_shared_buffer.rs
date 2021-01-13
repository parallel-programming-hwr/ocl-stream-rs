/*
 * opencl stream executor
 * Copyright (C) 2021 trivernis
 * See LICENSE for more information
 */

use crate::traits::ToOclBuffer;
use crate::utils::shared_buffer::SharedBuffer;
use ocl::{OclPrm, ProQue};

pub trait ToSharedBuffer<T>
where
    T: OclPrm,
{
    fn to_shared_buffer(&self, pro_que: &ProQue) -> ocl::Result<SharedBuffer<T>>;
}

impl<T> ToSharedBuffer<T> for Vec<T>
where
    T: OclPrm,
{
    fn to_shared_buffer(&self, pro_que: &ProQue) -> ocl::Result<SharedBuffer<T>> {
        let buffer = self.to_ocl_buffer(pro_que)?;

        Ok(SharedBuffer::new(buffer))
    }
}
