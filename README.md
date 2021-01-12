# Rust OpenCL Stream Executor

This crate provides abstractions over opencl execution to 
allow the streaming of results. This crate does not provide abstractions
over the ocl crate directly but allows to use the provided stream
executor to optimise the execution process.

## Usage

```rust
use crate::executor::OCLStreamExecutor;
use ocl::ProQue;

fn main() {
    // create the ProQue
    let pro_que = ProQue::builder()
        .src("
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
    
    // create the executor
    let stream_executor = OCLStreamExecutor::new(pro_que);

    // execute a closure that provides the results in the given stream
    let mut stream = stream_executor.execute_unbound(|ctx| {
        let pro_que = ctx.pro_que();
        let tx = ctx.sender();
        let input_buffer = pro_que.buffer_builder().len(100).fill_val(0u32).build()?;

        let kernel = pro_que
            .kernel_builder("bench_int")
            .arg(100)
            .arg(&input_buffer)
            .global_work_size(100)
            .build()?;
        unsafe {
            kernel.enq()?;
        }

        let mut result = vec![0u32; 100];
        input_buffer.read(&mut result).enq()?;

        for num in result {
            // send the results to the receiving thread
            tx.send(num)?;
        }

        Ok(())
    });

    let mut count = 0;

    // calculate the expected result values
    let num = (99f32.powf(2.0) + 99f32) / 2f32;
    // read the results from the stream
    while let Ok(n) = stream.next() {
        assert_eq!(n, num as u32);
        count += 1;
    }
    assert_eq!(count, 100)
}
```