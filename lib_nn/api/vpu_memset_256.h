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
 * @param     dst         Destination address
 * @param     src         Source address, must be word aligned.
 * @param     byte_count  Number of bytes to copy - may be zero
 */
void vpu_memset_256(void * dst, const void * src, unsigned int byte_count);

#endif
