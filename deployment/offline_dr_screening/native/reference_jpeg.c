#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <jpeglib.h>

#ifdef _WIN32
#define OFFIRST_EXPORT __declspec(dllexport)
#else
#define OFFIRST_EXPORT __attribute__((visibility("default")))
#endif

struct decode_state {
    struct jpeg_decompress_struct jpeg;
    struct jpeg_error_mgr error;
    jmp_buf recovery;
    uint8_t *pixels;
    int started;
};

static void fail_decode(j_common_ptr jpeg) {
    struct decode_state *state = (struct decode_state *)jpeg->client_data;
    longjmp(state->recovery, 1);
}

static void ignore_jpeg_message(j_common_ptr jpeg) {
    (void)jpeg;
}

// Pillow opens CMYK JPEGs with the inverted CMYK raw mode, then converts to RGB.
static uint8_t cmyk_channel(unsigned ink, unsigned black) {
    return (uint8_t)((ink * black + 127) / 255);
}

OFFIRST_EXPORT uint8_t *offirst_decode_jpeg(
    const uint8_t *data, size_t length, int32_t *width, int32_t *height
) {
    if (length > ULONG_MAX) return NULL;
    struct decode_state *state = calloc(1, sizeof(*state));
    if (!state) return NULL;
    state->jpeg.err = jpeg_std_error(&state->error);
    state->error.error_exit = fail_decode;
    state->error.output_message = ignore_jpeg_message;
    state->jpeg.client_data = state;
    if (setjmp(state->recovery)) goto fail;
    state->started = 1;
    jpeg_create_decompress(&state->jpeg);
    jpeg_mem_src(&state->jpeg, data, (unsigned long)length);
    if (jpeg_read_header(&state->jpeg, TRUE) != JPEG_HEADER_OK ||
        state->jpeg.data_precision != 8 ||
        (uint64_t)state->jpeg.image_width * state->jpeg.image_height > 40000000) goto fail;
    int cmyk = state->jpeg.jpeg_color_space == JCS_CMYK || state->jpeg.jpeg_color_space == JCS_YCCK;
    state->jpeg.out_color_space = cmyk ? JCS_CMYK : JCS_RGB;
    state->jpeg.do_fancy_upsampling = TRUE;
    state->jpeg.dct_method = JDCT_ISLOW;
    if (!jpeg_start_decompress(&state->jpeg) ||
        state->jpeg.output_components != (cmyk ? 4 : 3)) goto fail;

    const size_t row_size = (size_t)state->jpeg.output_width * 3;
    state->pixels = malloc(row_size * state->jpeg.output_height);
    if (!state->pixels) goto fail;
    JSAMPARRAY cmyk_row = cmyk ? (*state->jpeg.mem->alloc_sarray)(
        (j_common_ptr)&state->jpeg, JPOOL_IMAGE, state->jpeg.output_width * 4, 1
    ) : NULL;
    while (state->jpeg.output_scanline < state->jpeg.output_height) {
        JSAMPROW row = state->pixels + (size_t)state->jpeg.output_scanline * row_size;
        if (jpeg_read_scanlines(&state->jpeg, cmyk ? cmyk_row : &row, 1) != 1) goto fail;
        if (cmyk) {
            for (JDIMENSION x = 0; x < state->jpeg.output_width; ++x) {
                for (unsigned channel = 0; channel < 3; ++channel) {
                    row[x * 3 + channel] = cmyk_channel(cmyk_row[0][x * 4 + channel], cmyk_row[0][x * 4 + 3]);
                }
            }
        }
    }
    if (!jpeg_finish_decompress(&state->jpeg) || state->error.num_warnings != 0) goto fail;
    *width = (int32_t)state->jpeg.output_width;
    *height = (int32_t)state->jpeg.output_height;
    uint8_t *result = state->pixels;
    jpeg_destroy_decompress(&state->jpeg);
    free(state);
    return result;

fail:
    if (state->started) jpeg_destroy_decompress(&state->jpeg);
    free(state->pixels);
    free(state);
    return NULL;
}

OFFIRST_EXPORT void offirst_free_jpeg(uint8_t *pixels) {
    free(pixels);
}
