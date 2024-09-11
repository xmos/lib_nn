#include <stdint.h>

#ifndef _vpu_memset_256_h_
#define _vpu_memset_256_h_

/**
 * Function that replicates a vector. The source address must be word
 * aligned, the destination address is assumed to be aligned with the
 * replication pattern in the source. Any number of bytes can be copied.
 * There should not be an overlap between the destination and source.
 *
 * It is assumed that the source address contains 32 replicated bytes (if
 * the destination address is byte aligned), or that it contains 16
 * replicated shorts (if the destination address is 16-bit aligned), or
 * that it contains 8 replicated ints.
 *
 * broadcast_32_to_256() and BROADCAST_8_TO_32() cane be used to
 * create the source vector
 *
 * @param     dst         Destination address
 * @param     src         Source address, must be word aligned.
 * @param     byte_count  Number of bytes to copy - may be zero
 */
void vpu_memset_256(void *dst, const void *src, unsigned int byte_count);

/**
 * Function that replicates an int over a vector. The vector must be
 * aligned on an 8-byte boundary. In order to replicate a byte or short over
 * a vector, combine this with a call to BROADCAST_8_TO_32() or
 * BROADCAST_16_TO_32(). Declare the vector as a uint64_t x[] in order to
 * guarantee 8-byte alignement.
 *
 * @param     dst         Destination address, must be 8-byte aligned
 * @param     from        Value to be replicated
 */
void broadcast_32_to_256(void *dst, uint32_t from);

/**
 * Macro that replicates a byte over an int.
 * Use with broadcast_32_to_256() in order to replicate a byte over a vector
 */
#define BROADCAST_8_TO_32(f) (((uint8_t)f) * 0x01010101)

/**
 * Macro that replicates a short over an int
 * Use with broadcast_32_to_256() in order to replicate a short over a vector
 */
#define BROADCAST_16_TO_32(f) (((uint16_t)f) * 0x00010001)

/**
 * Macro that replicates a byte over a short
 */
#define BROADCAST_8_TO_16(f) (((uint8_t)f) * 0x00000101)

#endif
