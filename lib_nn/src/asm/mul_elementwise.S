	.text

	#define FUNCTION_NAME mul_elementwise_asm

	.cc_top FUNCTION_NAME.function,FUNCTION_NAME
	.align 4
BYTE_TO_WORD:
VECT8_ONES:
	.word    0x01010101
	.word    0x01010101
	.word    0x01010101
	.word    0x01010101

HALF_TO_WORD:
VECT16_ONES:
	.word    0x00010001
	.word    0x00010001
	.word    0x00010001
	.word    0x00010001
	.word    0x00010001
	.word    0x00010001
	.word    0x00010001
	.word    0x00010001

VECT_INT16_MAX:
	.word    0x7f807f80
	.word    0x7f807f80
	.word    0x7f807f80
	.word    0x7f807f80
	.word    0x7f807f80
	.word    0x7f807f80
	.word    0x7f807f80
	.word    0x7f807f80

VECT_INT8_MIN:
	.word    0x80808080
	.word    0x80808080
	.word    0x80808080
	.word    0x80808080

    .globl FUNCTION_NAME
    .globl FUNCTION_NAME.nstackwords
	.section	.cp.rodata.cst4,"aMc",@progbits,4
	

#define S_R4 0
#define S_R5 (S_R4 + 1)
#define S_R6 (S_R5 + 1)
#define S_R7 (S_R6 + 1)
#define S_R8 (S_R7 + 1)
#define S_R9 (S_R8 + 1)
#define S_R10 (S_R9 + 1)
#define S_MASK (S_R10 + 1)
#define S_TEMP (S_MASK + 1)
#define S_BIAS (S_TEMP + 8)
#define S_IN1_ZERO_POINT (S_BIAS + 8)
#define S_IN2_ZERO_POINT (S_IN1_ZERO_POINT + 4)
#define S_SCALAR (S_IN2_ZERO_POINT + 4) //place to put the result for vdepth8

#define NSTACKWORDS (S_SCALAR + 8)

	.text
	.issue_mode  dual
	.align	16
	.type	FUNCTION_NAME,@function

// void FUNCTION_NAME(
//	int8_t* in1_data, //r0
//	int8_t* in2_data, //r1
//	int element_count, //r2
//	nn_mul_params_t * params, //r3
//	int8_t * out_data); //sp[0]

#define in1_data r0 
#define in2_data r1 
#define element_count r2 
#define params r3
#define bias r4
#define scalar r5 
#define in1_zero_point r6 
#define in2_zero_point r7 
#define out_data r8 

#define temp3 r9
#define temp2 r10
#define temp r11

FUNCTION_NAME:
	dualentsp NSTACKWORDS

	std r4, r5, sp[0]
	std r6, r7, sp[1]
	std r8, r9, sp[2]
	stw r10, sp[6]

	ldap temp, BYTE_TO_WORD
	{ldw temp2, temp[0]; ldc temp, 0}
	{ld8u temp3, r3[temp]; ldc temp, 1}
	mul temp3, temp3, temp2 	//broadcast the byte to a word
	ldaw in1_zero_point, sp[S_IN1_ZERO_POINT]
	std temp3, temp3, in1_zero_point[0]
	std temp3, temp3, in1_zero_point[1]
	
	ld8u temp3, r3[temp]
	mul temp3, temp3, temp2 	//broadcast the byte to a word
	ldaw in2_zero_point, sp[S_IN2_ZERO_POINT]
	std temp3, temp3, in2_zero_point[0]
	std temp3, temp3, in2_zero_point[1]
	
	ldap temp, HALF_TO_WORD
	{ldw temp2, temp[0]; ldc temp, 1}
	{ld16s temp3, r3[temp]; ldc temp, 2}
	mul temp3, temp3, temp2 	//broadcast the short to a word
	ldaw bias, sp[S_BIAS]
	std temp3, temp3, bias[0]
	std temp3, temp3, bias[1]
	std temp3, temp3, bias[2]
	std temp3, temp3, bias[3]

	{ld16s temp3, r3[r11];ldc temp, 0}
	mul temp3, temp3, temp2 	//broadcast the short to a word
	ldaw scalar, sp[S_SCALAR]
	std temp3, temp3, scalar[0]
	std temp3, temp3, scalar[1]
	std temp3, temp3, scalar[2]
	std temp3, temp3, scalar[3]

	stw temp, sp[S_MASK]

	//load out_data with sp[NSTACKWORDS+1]
	ldw out_data, sp[NSTACKWORDS+1]
	ldc temp, 512
	FUNCTION_NAME.loop:
		
		{vsetc temp; nop}
		{vclrdr; ldap temp, VECT8_ONES}
	
		{vldc temp[0] ; nop}
		{vlmacc in1_data[0] ; nop}
		{vlmacc in2_data[0] ; nop}
		{vlmacc in1_zero_point[0] ; nop}
		{vlmacc in2_zero_point[0] ; ldc temp, 16}

		{vldc in1_data[0] ; add in1_data, in1_data, temp}
		{vlmacc in2_zero_point[0] ; nop}
		{vlmacc in2_data[0] ; nop}

		{vldc in1_zero_point[0] ; ldc temp2, 1}
		{vlmacc in2_data[0] ; add in2_data, in2_data, temp}
		{vlmacc in2_zero_point[0] ; shl temp, temp2, 8}

		{vsetc temp; ldap temp, VECT16_ONES}
		
		vlsat temp[0] //shift all right by one and truncate to 16 bits per element

		vlmul scalar[0]
		{vladd bias[0]; ldaw temp2, sp[S_TEMP]}
		{vstr temp2[0] ; ldc temp, 3}
		
		ld16s temp, r3[temp]
		
		vlashr temp2[0], temp

		{vstr temp2[0] ; ldap temp, VECT_INT16_MAX }  
			
		{vladd temp[0]                    ; mkmsk temp2, 2}
		{vdepth1                          ; ldaw temp3, sp[S_MASK]}

		vstrpv temp3[0], temp2

		ldap temp, VECT_INT8_MIN
		{vldr temp[0]                     ; ldaw temp, sp[S_TEMP] } 

		{mkmsk temp2, element_count       ; ldw temp3, sp[S_MASK] }
		{zext temp2, 16 ; nop}
		vstrpv out_data[0], temp2         //store the int8_mins to output Y

		{vldr temp[0]                     ; ldc temp, 16}
		{vdepth8                          ; andnot temp2, temp3}
		vstrpv out_data[0], temp2

		{ldc temp2, 1; sub element_count, element_count, temp}
		{add out_data, out_data, temp; lss temp2, element_count, temp2}
		{bf temp2, FUNCTION_NAME.loop; shl temp, temp, 5}

	ldd r4, r5, sp[0]
	ldd r6, r7, sp[1]
	ldd r8, r9, sp[2]
	ldw r10, sp[6]
	retsp NSTACKWORDS

	.cc_bottom FUNCTION_NAME.function
	.set	FUNCTION_NAME.nstackwords,NSTACKWORDS
	.set	FUNCTION_NAME.maxcores,1
	.set	FUNCTION_NAME.maxtimers,0
	.set	FUNCTION_NAME.maxchanends,0
.Ltmp2:
	.size	FUNCTION_NAME, .Ltmp2-FUNCTION_NAME
