// Embeds nnue/net/hydray.nnue and the eval-scale reference nnue/scale_ref.txt
// into the binary with the assembler's .incbin, for the Linux and mingw builds alike. The path is relative to the compiler's
// working directory, the repo root; the makefile declares the dependency,
// which -MMD cannot track.

__asm__(
    ".section .rodata\n"
    ".balign 64\n"
    ".global g_hydrayEmbeddedNetStart\n"
    "g_hydrayEmbeddedNetStart:\n"
    ".incbin \"nnue/net/hydray.nnue\"\n"
    ".global g_hydrayEmbeddedNetEnd\n"
    "g_hydrayEmbeddedNetEnd:\n"
    ".global g_hydrayScaleRefStart\n"
    "g_hydrayScaleRefStart:\n"
    ".incbin \"nnue/scale_ref.txt\"\n"
    ".global g_hydrayScaleRefEnd\n"
    "g_hydrayScaleRefEnd:\n"
    ".text\n"
);
