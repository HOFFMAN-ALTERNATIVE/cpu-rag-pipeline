# Running language models on CPU

## Quantization
Quantizing weights to 4 or 8 bits shrinks memory use several times and speeds up inference
on CPUs, at a small cost in output quality.

## Threads and memory bandwidth
Token generation on CPU is usually limited by memory bandwidth rather than raw compute.
Setting the thread count to the number of physical cores is a good starting point.

## Keeping the model warm
Loading a multi-gigabyte model from disk takes seconds, so long-running servers keep the
model resident in RAM and process requests from a queue.
