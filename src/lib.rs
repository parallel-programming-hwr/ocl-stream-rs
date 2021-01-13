/*
 * opencl stream executor
 * Copyright (C) 2021 trivernis
 * See LICENSE for more information
 */

pub mod executor;
pub mod traits;
pub mod utils;

pub use executor::stream;
pub use executor::OCLStreamExecutor;
// reexport the ocl crate
pub use ocl;

#[cfg(test)]
mod tests {
    use crate::executor::OCLStreamExecutor;
    use crate::traits::*;
    use crate::utils::shared_buffer::SharedBuffer;
    use ocl::ProQue;
    use std::ops::Deref;

    #[test]
    fn it_streams_ocl_calculations() {
        let pro_que = ProQue::builder()
            .src(
                "\
        __kernel void bench_int(const uint limit, __global int *NUMBERS) {
            uint id = get_global_id(0);
            int num = NUMBERS[id];
            for (int i = 0; i < limit; i++) {
                num += i;
            }
            NUMBERS[id] = num;
        }",
            )
            .dims(1)
            .build()
            .unwrap();
        let stream_executor = OCLStreamExecutor::new(pro_que);
        let input_buffer: SharedBuffer<u32> = vec![0u32; 100]
            .to_shared_buffer(stream_executor.pro_que())
            .unwrap();

        let mut stream = stream_executor.execute_bounded(10, move |ctx| {
            let pro_que = ctx.pro_que();
            let tx = ctx.sender();

            let kernel = pro_que
                .kernel_builder("bench_int")
                .arg(100)
                .arg(input_buffer.inner().lock().deref())
                .global_work_size(100)
                .build()?;
            unsafe {
                kernel.enq()?;
            }

            let mut result = vec![0u32; 100];
            input_buffer.read(&mut result)?;

            for num in result {
                tx.send(num)?;
            }

            Ok(())
        });

        let mut count = 0;

        let num = (99f32.powf(2.0) + 99f32) / 2f32;
        while let Ok(n) = stream.next() {
            assert_eq!(n, num as u32);
            count += 1;
        }
        assert_eq!(count, 100)
    }
}
