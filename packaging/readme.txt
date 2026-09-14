gpusqz compresses and decompresses files on the GPU.

  gpusqz c <input> <output.gsz>     compress
  gpusqz d <input.gsz> <output>     decompress
  gpusqz devices                    list the GPUs gpusqz can use

NVIDIA GPUs use CUDA; AMD, Apple, Intel and other GPUs use Vulkan.
gpusqz_refdec decodes .gsz files on the CPU, without a GPU.

The installer puts both programs on your PATH. Open a new terminal
after installing, then run "gpusqz devices".
