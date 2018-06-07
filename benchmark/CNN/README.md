# CNN BenchMark


## 1. How to run

> 1. At first, you should parse the caffe model with [Anakin Parser(Converter)](../../docs/Manual/Converter_en.md).
> 2. Switch to *source_root/benchmark/CNN* directory. Use 'mkdir ./models' to create ./models and put anakin models into this file.
> 3. Use command 'sh run.sh', we will create files in logs to save model log with different batch size. Finally, model latency summary will be displayed on the screen.
> 4. If you want to get more detailed information with op time, you can modify CMakeLists.txt with setting `ENABLE_OP_TIMER` to `YES`, then recompile and run. You will find detailed information in  model log file.

# RNN BenchMark

Not Support Yet.
