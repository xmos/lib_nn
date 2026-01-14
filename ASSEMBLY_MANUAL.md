# XS3 Assembly Optimisation Reference

Based on **“The XMOS XS3 Architecture”** (Document **XM-014007-PS v2.0.0**, publication date **2024-10-06**).

This document is written as a **single self-contained markdown reference** aimed at covering **~99% of the patterns you use in hand-written, optimised XS3 assembly**: the execution model, dual-issue/lanes, instruction fetch behaviour, core instruction families, resources (channels/ports/timers/locks), events/interrupts/exceptions, and the FP/vector units—with examples.

---

## 1. What is “XS3” in practice

xcore.ai devices contain multiple **XCORE processors**, each with its own memory, connected by an on-chip switch and (optionally) off-chip XLINKs. XS3 evolves XS2 mainly by adding a **vector unit**, **floating point**, and **external memory support** (including “software-defined memory”).

---

## 2. Execution model: hardware threads, pipeline, fetch, “virtual cycles”

### 2.1 Threads as predictable “virtual processors”

- Instructions are issued from **runnable threads round-robin**, skipping threads that are unused or paused (e.g., waiting for I/O).
- With **n runnable threads**, you can treat them like **n virtual processors** at **≥ 1/n** of the core clock rate—except when **n < pipeline depth p**, in which case the per-thread rate is **≤ 1/p**.
- Restart latency for a waiting thread is bounded: re-start is always **≤ 1 thread cycle** (subject to the pipeline latency after re-issue).

### 2.2 Per-thread instruction buffer (critical for optimisation)

Each thread has a **256-bit instruction buffer** holding **16 short (16-bit) instructions** or **8 long (32-bit) instructions**.

### 2.3 The “one memory stage” rule (why loads/stores can force bubbles)

The pipeline has a **memory access stage** used for _both_ data access and instruction fetch. Rules:

- Instructions needing **data access** use the memory stage for that access.
- **Branches** fetch the branch target during the memory stage **unless** they also require a data access (then they leave the buffer empty).
- **Conditional branches** only ever fetch instructions around the target address.
- **ALU/non-memory** instructions use the memory stage to **fetch instructions** into the thread’s instruction buffer (if not full).
- If the buffer is empty when an instruction should issue, the core issues a **fetch no-op**, whose memory stage refills the buffer.

**Optimisation implication:** long runs of loads/stores can starve instruction fetch. Break them up with ALU ops so fetch can happen.

### 2.4 Fast-mode (for 1 I/O per “virtual cycle”)

Normally, if an I/O instruction can’t complete (no data / no space), the thread is **paused** and later **restarted**, typically by **re-issuing** the instruction.

In **fast-mode**, a thread is **not descheduled** when an instruction can’t complete; the instruction is **re-issued until it completes**, enabling “one input/output per virtual cycle” behaviour.

### 2.5 Events/interrupts: event no-op and “empty buffer” effect

For events/interrupts, the vector must be supplied and a target fetch must occur. The same ready-request mechanism is used, but the thread becomes runnable with an **empty instruction buffer**. An **event no-op** is used so the resource can provide the vector in time for fetch. Bounded response: at most **1 virtual cycle** to process the vector, and at most **2 virtual cycles** before instruction issue after an event/interrupt.

---

## 3. Registers you actually care about (scalar + control + vector)

### 3.1 Scalar register sets

- **Operand registers:** `r0` … `r11` (12 total).
- Common “pointer-ish” registers used by many instructions:
  - `sp` stack pointer
  - `dp` data pointer
  - `cp` constant pool pointer
  - `lr` link register
  - `pc` program counter (control)
  - `sr` status register (control)
- Exception/interrupt-related control registers:
  - `spc` saved program counter
  - `ssr` saved status register
  - `et` exception type
  - `ed` exception data
  - `sed` (secondary exception data)
  - `kep` kernel entry pointer
  - `ksp` kernel stack pointer

### 3.2 Status register bits that matter for performance

Key `sr` bits include:

- `DI`: dual-issue enable (per-thread; set/cleared per function via `DUALENTSP` / `ENTSP`, restored by `RETSP`).
- `HIPRI`: high-priority scheduling.
- `FAST`: fast-mode behaviour for I/O waits.
- `EEBLE` / `IEBLE`: events/interrupts enable bits.
- `KEDI`: controls whether dual-issue is enabled in kernel mode on entry.
- Other bits exist (e.g., waiting/sink indicators).

### 3.3 Vector registers

The vector unit provides three vector registers and a control register:

- `vC` (typically coefficients / constants)
- `vD` (typically data / headroom accumulator storage depending on op)
- `vR` (typically result / accumulator low part)
- `vCTRL` (32-bit): type/shift/magnitude configuration

`vCTRL` fields:

- bits `[31:28]` magnitude (`vctrl_m`), 0 = empty
- bits `[27:24]` shift mode (`VEC_SHR`, `VEC_SH0`, `VEC_SHL`)
- bits `[23:20]` type (`VEC_INT_32`, `VEC_INT_16`, `VEC_INT_8`)
- bits `[19:0]` reserved (0)

---

## 4. Instruction encoding basics that affect optimisation

### 4.1 Issue width and PC semantics

- `iw` = issue width in bytes: **2 in single-issue**, **4 in dual-issue**. Branch offsets are scaled by `iw`.
- When `pc` is used by an instruction, it points to the **next instruction**; `pc_old` refers to the current instruction’s location.

### 4.2 16-bit vs 32-bit instructions, and prefixes

- Many instructions have `u16` or `u20` immediates. If the **top 10 bits** of such an operand are non-zero, a **16-bit `PFIX` prefix** is used to encode them; low bits are encoded in the instruction.
- An **`EOPR` prefix** encodes “more than 3 operands” forms or less common instructions.

**Optimisation implication:** instruction density and fetch pressure matter because the memory stage is shared between fetch and loads/stores. Long-immediate-heavy code can increase fetch demand (more instruction words).

---

## 5. Dual-issue: lanes, alignment rules, hazards

### 5.1 The dual-issue model

XS3 has two pipeline lanes:

- **Memory lane:** all memory instructions, branches, and basic arithmetic.
- **Resource lane:** all resource instructions and basic arithmetic.

Each thread may execute in **dual-issue mode**, allowing:

- **two 16-bit instructions** in one thread cycle (one per lane), or
- **one 32-bit instruction** in one thread cycle.

Dual-issue is controlled by the `DI` bit in `sr`.

### 5.2 Alignment requirements in DI mode

In dual-issue:

- 32-bit instructions must be **32-bit aligned**.
- Pairs of 16-bit instructions must be aligned on a **32-bit boundary**.
- `pc` is always **32-bit aligned** and points to an **issue slot**, not an individual instruction.
- Slot layout:
  - addresses `4n+0..4n+1`: **resource-lane** 16-bit instruction
  - addresses `4n+2..4n+3`: **memory-lane** 16-bit instruction
  - long (32-bit) instruction occupies `4n+0..4n+3`

### 5.3 Register write hazard rule (easy way to crash yourself)

If two instructions execute simultaneously, their **destination operands must be disjoint**; otherwise an **exception** is raised.

### 5.4 Stall/exception interactions across lanes

- If the **resource lane stalls** a thread, the **other lane stalls too**.
- On interrupt/exception in DI mode: no registers are overwritten, and the PC points to the instruction to be re-executed.
- If one lane faults, that exception is reported and the other lane’s instruction is aborted; some in-progress memory stores may still complete.

### 5.5 DI enable is “per function”

- `DI` is saved in the **lowest bit of `lr`** on a function call and restored by `RETSP`.
- `DUALENTSP x` sets DI for the function; `ENTSP x` clears it.

### 5.6 Lane placement cheat-sheet (practical subset)

- **Memory-lane-only** includes most loads/stores and branches (`LDW*`, `STW*`, `LDD*`, `STD*`, `BR*`, `BL*`, `RETSP`, cache ops, and vector ops like `VLADD`, `VLMUL`, `VLMACC`, etc.).
- **Resource-lane-only** includes channel/port/timer/lock/resource ops (`IN`, `OUT`, `OUTCT`, `INCT`, `GETR`, `FREER`, `SETV`, `TSTART`, etc.).

### 5.7 Example: designing a DI “issue slot”

You often want each issue slot to do:

- **Memory lane:** a load/store
- **Resource lane:** pointer arithmetic or a resource op

Example (illustrative structure):

```asm
# Issue slot N:
#   Resource lane: ADD   r1, r1, Bpw      # advance pointer
#   Memory   lane: LDW   r0, r1[0]        # load word
# Rule: destination regs disjoint (LDW writes r0, ADD writes r1)
```

This style both (a) uses both lanes and (b) breaks up load/store streaks with ALU ops, helping instruction fetch.

---

## 6. Memory model, alignment, and high-throughput load/store patterns

### 6.1 Memory fundamentals

- Memory is **byte addressed**, **word aligned**, **little endian**.
- Unaligned memory accesses raise **`ET_LOAD_STORE`** (e.g., word load/store where the low `log2(Bpw)` bits aren’t zero, or misaligned 16-bit access for `LD16S`/`ST16`).
- Many instructions scale offsets by `Bpw` to improve encoding density; if `sp/cp/dp` is ever made unaligned, subsequent scaled accesses can fault.

### 6.2 The “core” scalar load/store instructions

Common word access groups:

- Stack relative: `LDWSP`, `STWSP`, `LDAWSP`
- Data pointer relative: `LDWDP`, `STWDP`, `LDAWDP`
- Constant pool / program address helpers: `LDC`, `LDWCP`, `LDAWCP`, `LDWCPL`, `LDAPF`, `LDAPB`
- Base+offset addressing: `LDW`, `STW`, `LDAW` variants (scaled word offsets; register or small immediate forms)

Byte/halfword access:

- `LD8U` (zero-extend byte)
- `LD16S` (signed 16-bit)
- `ST8`, `ST16`

Double-word (great for fast save/restore):

- `LDD` / `STD` (load/store two words)
- `LDDSP` / `STDSP` (stack-based double)
- `STSPC`, `STSSR`, `LDSPC`, `LDSSR` for saved PC/SR access

### 6.3 Avoiding fetch no-ops in load/store-heavy code

Because instruction fetch shares the memory stage, long runs of memory ops can empty the buffer and force fetch no-ops. A common fix: interleave address arithmetic or independent ALU ops between memory ops.

### 6.4 External memory, cache control, and software-defined memory

XS3 supports:

- **External memory** accessed through a cache (optional), with cache control ops: `FLUSH`, `INVALIDATE`, `PREFETCH`.
- **Software-defined memory (SWMEM)**: accesses to a SWMEM address raise a memory exception and are handled in software. A typical SWMEM handler uses `IN` to retrieve the address from the SWMEM resource and then services reads/writes via memory + `OUT` + `SETC START`.

Notes:

- External memory has **much lower performance** than internal memory, and performance can degrade if multiple threads access external memory simultaneously.

---

## 7. Branching, calling, stack: how to structure fast functions

### 7.1 Branch families (relative, scaled by issue width)

Branch instructions use `iw` scaling (2 single-issue, 4 dual-issue).
Absolute/indirect forms exist (`BAU`, etc.), and branch/call forms can fault with `ET_ILLEGAL_PC` if target is invalid/misaligned.

### 7.2 Calling and return: `ENTSP`, `DUALENTSP`, `RETSP`

- `ENTSP n`: saves `lr` to `sp[0]` and extends stack frame by `n` words; also clears DI for the function.
- `DUALENTSP n`: like `ENTSP`, but enables DI for the function.
- `RETSP n`: restores stack pointer, loads `sp[0]` to `pc`, and restores DI from the saved bit in `lr`.

### 7.3 ABI notes (toolchain interoperability)

The XMOS 32-bit ABI specifies:

- First four registers (`r0–r3`) pass parameters; extras go on stack.
- Returns use `r0–r3` similarly.
- `r0–r3` caller-save; `r4–r10` callee-save; `r11` caller-save scratch.
- `cp`, `dp`, `sp`, `lr` callee-save.

### 7.4 Example: minimal scalar function (single-issue)

```asm
# int add2(int a, int b)
# ABI: a in r0, b in r1; return in r0
add2:
    ENTSP   0          # save lr to sp[0], no locals
    ADD     r0, r0, r1 # r0 = a + b
    RETSP   0
```

### 7.5 Example: dual-issue leaf with a small stack frame

```asm
# int sum_words(const int *p, unsigned n)
# r0=p, r1=n; returns sum in r0
sum_words:
    DUALENTSP  2            # enable DI; allocate 2 words
    STWSP      r4, 1        # spill callee-saved if used
    STWSP      r5, 2

    LDC        r4, 0        # sum=0

loop:
    # slot: (conceptual)
    #   R lane:  ADD   r0, r0, Bpw       # advance pointer
    #   M lane:  LDW   r5, r0[-1]        # load previous word (illustrative)

    ADD        r4, r4, r5   # sum += loaded
    SUB        r1, r1, 1    # n--
    SETCI      cond, r1     # set condition based on r1
    BRBF       loop, <back> # loop while n > 0 (pattern)

    MOV        r0, r4       # return sum

    LDWSP      r4, 1
    LDWSP      r5, 2
    RETSP      2
```

---

## 8. Conditions and conditional control-flow (what “cond” is used for)

The ISA uses a `cond(...)` predicate used by conditional branches and other conditional behaviour (e.g. `SETCI` sets a condition from a comparison). A practical loop pattern: decrement a counter register, set a condition from it, branch based on that condition.

---

## 9. Resources: allocation, ownership, and the “resource dependency” trap

### 9.1 Resources are per-thread owned

Ports, channel ends, timers, locks, etc. are **resources**. You allocate them (e.g., with `GETR`) and they become owned/used by a thread.

### 9.2 ET_RESOURCE_DEP: “don’t touch the same resource within 4 cycles from multiple threads”

If multiple threads access the **same resource** within **4 cycles** of each other, an **ET_RESOURCE_DEP** exception is raised.

---

## 10. Channels and interconnect: fast, deterministic message passing

### 10.1 XLINK tokens (what can be sent)

XLINK transports **data tokens (bytes)** and **control tokens**. The token space is partitioned:

- 0–127: application tokens
- 128–191: special tokens (architecturally defined)
- 192–223: privileged tokens (require privilege; otherwise exception)
- 224–255: hardware tokens (cannot be output by software; output attempt causes exception)

`END` and `PAUSE` are special control tokens that end routes through the interconnect; `END` is delivered to the receiver; `PAUSE` disconnects the route but is not delivered to the receiving thread.

### 10.2 Channel end setup and network selection

- Each channel end has a **destination register**, initialised via `SETD`; readable with `GETD`.
- Channels can be assigned to independent networks with `SETN` and queried with `GETN`.

### 10.3 Core channel instructions you use constantly

- Data words: `OUT`, `IN` (IN traps on control token)
- Token/byte: `OUTT`, `INT` (INT traps on control token)
- Control: `OUTCT`, `OUTCTI`, `INCT`
- Checks/peek: `CHKCT`, `CHKCTI`, `TESTCT`, `TESTWCT`

Buffering:

- Output when full → pause until space.
- Input when insufficient → pause until enough data.

### 10.4 Connections, END, shared destinations

- Connection established on first output; persists until **END** is sent.
- Destination can be shared; served round-robin; other senders queue until END.
- `TESTLCL` tests whether a destination is local.

### 10.5 Example: fixed-size message with END checking

```asm
# Sender: send two words then END
    OUT     c, r0
    OUT     c, r1
    OUTCTI  c, END

# Receiver: read two words and verify END
    IN      r2, c
    IN      r3, c
    CHKCTI  c, END
```

### 10.6 Example: synchronized request/response handshake

```asm
# Sender (sync)
    OUT     c, r0
    OUTCTI  c, END
    CHKCTI  c, END     # waits for receiver's END ack

# Receiver (sync)
    IN      r1, c
    CHKCTI  c, END     # received message end
    OUTCTI  c, END     # ack
```

---

## 11. Locks: mutual exclusion using IN/OUT

Locks are explicit resources:

- Allocate: `GETR l, LOCK`
- Claim: `IN` on the lock (waits if busy)
- Free (release): `OUT` to the lock (data ignored)
- Free resource: `FREER l`

### Example: critical section with a lock

```asm
    GETR    r0, LOCK      # r0 = lock resource id
    IN      r0, r0        # claim lock (waits if busy)

    # --- critical section ---
    # ... protected operations ...
    # ------------------------

    OUT     r0, r1        # release lock (data r1 ignored)
    FREER   r0
```

---

## 12. Timers and clocks: precise timing without jitter

### 12.1 Timers

- Allocate: `GETR t, TIMER`
- `GETTIME` reads current time
- `SETC` timer mode:
  - `UNCOND`: always ready; `IN` returns immediately
  - `AFTER`: ready when time is after its DATA; `IN` waits until then
- `SETD` sets the DATA (deadline)

#### Example: delay-until deadline

```asm
    GETR    r0, TIMER
    SETC    r0, AFTER

    GETTIME r1            # r1 = now
    ADD     r1, r1, 1000  # deadline = now + 1000 ticks (example)
    SETD    r0, r1

    IN      r2, r0        # waits until timer >= deadline; returns current time
    FREER   r0
```

### 12.2 Clocks (for port clocking)

- `SETCLK` connects a clock resource to a source.
- Divider set via `SETD` low 8 bits; `n` yields output frequency `f / 2^n`.
- Start/stop with `SETC c, START` / `SETC c, STOP`.

---

## 13. Ports and high-speed I/O: IN/OUT, INSHR/OUTSHR, buffering

Ports are resources interfacing to physical pins.

### 13.1 Base port transfer semantics

- Port has a transfer register.
- `IN` reads it (zero-extended).
- `OUT` writes low bits to it.

### 13.2 Shift-optimised port ops (use for serial protocols)

- `INSHR`: shift destination reg right, fill high bits with bits from port.
- `OUTSHR`: output LSBs from reg and shift reg (throughput tool).

### Example: shift out 32 bits (SPI-like skeleton)

```asm
    OUTSHR   p, r0
    OUTSHR   p, r0
    OUTSHR   p, r0
    OUTSHR   p, r0
```

---

## 14. Events, interrupts, exceptions: low-latency multiplexing and fault handling

### 14.1 Events vs interrupts (conceptual)

- Each resource has a vector register and event enable bit, configured by `SETV` and `SETEV`.
- When a resource becomes ready, it can generate an event/interrupt.
- If events enabled in `sr`, the thread restarts at the resource vector.

### 14.2 Key event/interrupt instructions

- `SETV`, `SETEV`, enable/disable event ops (`EEU`, `EET`, `EEF`, `EDU`)
- `WAITEU`, `WAITET`, `WAITEF`, `CLRE`, `GETED`

### 14.3 Interrupt entry/exit model (kernel entry)

On interrupt:

- save `pc`/`sr` to `spc`/`ssr`
- dual-issue set per `KEDI`
- events/interrupts disabled
- `pc = kep`, `sp = ksp`
- return with `KRET`

### 14.4 Exception essentials

On exception:

- `pc`/`sr` saved
- `et`/`ed` set
- handler at `kep`, return with `KRET`

Common exception types:

- `ET_LOAD_STORE`, `ET_ILLEGAL_RESOURCE`, `ET_RESOURCE_DEP`, `ET_LINK_ERROR`,
  `ET_ARITHMETIC`, `ET_ILLEGAL_PC`, `ET_ILLEGAL_INSTRUCTION` (and `ET_IOLANE` ORed for resource-lane faults)

---

## 15. Specialised scalar instructions (fast primitives)

Useful primitives include:

- Division/remainder: `DIVS`, `DIVU`, `REMS`, `REMU`, `LDIVU`
- MAC/saturation/extract: `MACCS`, `MACCU`, `MACCU.1`, `LSATS`, `LEXTRACT`, `LINSETR`
- CRC: `CRC8`, `CRC32`

---

## 16. Floating point (32-bit IEEE) essentials

FP ops include:

- `FADD`, `FSUB`, `FMUL`, `FMACC`
- `FMANT`, `FSEXP`, `FMAKE`, `FENAN`
- Comparisons: `FGT`, `FLT`, `FEQ`, `FUN`

Rounding mode is in the **2 LSBs of `sr`** (set via `SETSR`/`CLRSR`).

### Example: fused multiply-add

```asm
    FMACC   r0, r1, r2    # r0 = a*b + c
```

---

## 17. Vector unit: how to get performance

### 17.1 Configuration: vCTRL

Vector registers: `vC`, `vD`, `vR`, control `vCTRL`.

`vCTRL` fields:

- magnitude `[31:28]`
- shift mode `[27:24]` (`VEC_SHR`, `VEC_SH0`, `VEC_SHL`)
- type `[23:20]` (`VEC_INT_32`, `VEC_INT_16`, `VEC_INT_8`)

### 17.2 Config ops

- `VSETC` configures `vCTRL` from `r11`
- `VGETC` reads `vCTRL` into `r11`

### 17.3 Vector memory ops

- Loads: `VLDC`, `VLDD`, `VLDR`
- Stores: `VSTC`, `VSTD`, `VSTR`, `VSTRPV`

### 17.4 Arithmetic / complex / FFT / MACC

- Elementwise: `VLADD`, `VLSUB`, `VLMUL`
- Complex mult: `VCMR`, `VCMI`, `VCMCR`, `VCMCI`
- FFT: `VLADSB`, `VFTFF`, `VFTFB`, `VFTTF`, `VFTTB`
- MACC: `VCLRDR`, `VLMACC`, `VLMACCR`, `VLMACCR1`
- Reduce: `VLSAT`, `VADDDR`

### Example: inner-product skeleton

```asm
    VSETC     r11          # configure vCTRL
    VCLRDR                 # clear accumulators
    VLDC      [r0]         # load coefficients into vC
    VLMACCR   [r1]         # accumulate inner product into (vD,vR)
    VLSAT                 # reduce accumulator to normal vector (conceptual)
```

---

## 18. Practical optimisation checklist

### 18.1 Scheduling and fetch

- Avoid emptying the instruction buffer: break long memory-op runs with ALU ops.
- Prefer dense 16-bit encodings when possible to reduce fetch pressure.

### 18.2 Dual-issue effectiveness

- Use `DUALENTSP` for hot code; respect DI alignment and pairing rules.
- Ensure destination regs disjoint when pairing instructions.
- Pair memory ops with pointer arithmetic/resource ops when possible.

### 18.3 Resource correctness

- Avoid cross-thread tight sharing of the same resource (ET_RESOURCE_DEP).
- Use `END`/`PAUSE` correctly for channels.
- Use locks for shared state when required.

### 18.4 Vector/FP performance

- Keep `vC` resident, stream data into MACC ops, reduce at end.
- Use `VFT*` family for FFT.
- Use `FMACC` for float fused operations.

---

## 19. Appendix: common failure modes

- Misaligned access → `ET_LOAD_STORE`.
- Non-disjoint destinations in DI → exception.
- Illegal channel token/destination → `ET_LINK_ERROR`.
- Same resource used across threads too closely → `ET_RESOURCE_DEP`.
- Invalid vector alignment for event handlers → `ET_ILLEGAL_PC`.
