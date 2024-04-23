#ifndef _vpu_memmove_word_aligned_h_
#define _vpu_memmove_word_aligned_h_

/**
 * Function that copies a block of memory. Both source and destination
 * address must be word aligned. Any number of bytes can be copied. There
 * may be an overlap between the destination and source.
 *
 * @param     dst         Destination address, must be word aligned.
 * @param     src         Source address, must be word aligned.
 * @param     byte_count  Number of bytes to copy - may be zero
 */
void vpu_memmove_word_aligned(void * dst, const void * src, unsigned int byte_count);

#endif
