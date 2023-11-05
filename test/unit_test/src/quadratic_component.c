
int main(void) {
    quadratic_activiation_function_t output;
    for(int chunks = 16; chunks < 2049; chunks *= 2) {
        approximate_activation_function(AV_TANH, 8.0/32768, 1.0/32768, chunks, &output);
        if (chunks == 128) {
           printf("int64_t coeffs[] = {\n");
           for(int i = 0; i < chunks; i++) {
               printf("    0x%016llxLL,\n", *(int64_t *)&output.coefficients[i]);
           }
           printf("};\n");
        }
    }
}


