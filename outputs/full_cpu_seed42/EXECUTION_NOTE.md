This is an interrupted CPU experiment, not the primary study result.

The restricted process reported `torch.cuda.is_available() == False`, so initial
execution used CPU. A local notebook kernel subsequently exposed a GeForce GTX
1660 Ti (6 GB) outside the restriction. The CPU process was stopped after its
second random-negative training epoch, preserving all artifacts. The full study
was restarted as `outputs/full_gpu_seed42/`, with GPU acceleration and SciBERT
pair models. Do not present this directory as a completed full comparison.
