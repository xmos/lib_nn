Lookup8 Simple Example
======================

This example demonstrates a minimal use of ``lookup8()``.

It applies a full 256-entry look-up table to an ``int8`` vector.

The LUT implements:

* double: ``y = 2*x`` with int8 saturation

The LUT is embedded directly in the C source as a static ``int8_t[256]`` array
and was generated with Python.

The code uses one clear call:

* ``lookup8(output_u8, input_u8, double_lut, 0, SIZE)``

All data paths use ``uint8_t``, matching the ``lookup8()`` API directly.

Why this is useful:
LUT mapping is a fast way to implement per-element transforms without branching.
The same pattern can be reused for clipping, custom activation, calibration,
gamma correction, and quantized nonlinear remaps.

Build and run from this directory:

.. code-block:: console

    cmake -G "Unix Makefiles" -B build
    cmake --build build
    xsim bin/lookup8.xe
