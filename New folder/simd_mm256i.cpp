#include <iostream>
#include <vector>
#include <fstream>
#include <inttypes.h>
#include <malloc.h>
//#include <pmmintrin.h>
#include <immintrin.h>

using namespace std;

#pragma pack(push,1)
struct BMPFileHeader {
    uint16_t file_type{ 0x4D42 };          // BMP file always have first 2 bytes 4D42
    uint32_t file_size{ 0 };               // Size of the file (in bytes)
    uint16_t reserved1{ 0 };               
    uint16_t reserved2{ 0 };               
    uint32_t offset_data{ 0 };             // Start position of pixel data
};

struct BMPInfoHeader {
    uint32_t size{ 0 };                      // Size of BMP Info Header
    int32_t width{ 0 };                      // width of bitmap in pixels
    int32_t height{ 0 };                     // width of bitmap in pixels
                                             //       (if positive, bottom-up, with origin in lower left corner)
                                             //       (if negative, top-down, with origin in upper left corner)
    uint16_t planes{ 1 };                    // No. of planes for the target device, this is always 1
    uint16_t bit_count{ 0 };                 // No. of bits per pixel
    uint32_t compression{ 0 };               // 0 or 3 - uncompressed. THIS PROGRAM CONSIDERS ONLY UNCOMPRESSED BMP images
    uint32_t size_image{ 0 };                // 0 - for uncompressed images
    int32_t x_pixels_per_meter{ 0 };
    int32_t y_pixels_per_meter{ 0 };
    uint32_t colors_used{ 0 };               // No. color indexes in the color table. Use 0 for the max number of colors allowed by bit_count
    uint32_t colors_important{ 0 };          // No. of colors used for displaying the bitmap. If 0 all colors are required
};

struct BMPColorHeader {
    uint32_t red_mask{ 0x00ff0000 };         // Bit mask for the red channel
    uint32_t green_mask{ 0x0000ff00 };       // Bit mask for the green channel
    uint32_t blue_mask{ 0x000000ff };        // Bit mask for the blue channel
    uint32_t alpha_mask{ 0xff000000 };       // Bit mask for the alpha channel
    uint32_t color_space_type{ 0x73524742 }; // Default "sRGB" (0x73524742)
    uint32_t unused[16]{ 0 };                // Unused data for sRGB color space
};
#pragma pack(pop)

struct BMP 
{
    // Header imagine BMP
    BMPFileHeader  file_header;
    BMPInfoHeader  bmp_info_header;
    BMPColorHeader bmp_color_header;
    
    //Data array

    union simd_uint8_t
    {
        __m256i a_simd;
        short a[16];
    };

    vector<simd_uint8_t> data_simd_b;
    vector<simd_uint8_t> data_simd_g;
    vector<simd_uint8_t> data_simd_r;
   
    // Image data
    vector<uint8_t> data;

    void readBMP(const char* filename)
    {
        ifstream inp{filename, ios_base::binary};
        
        // Check if inp exist and image open is a BMP file 
        check(inp);
        // Look for start pixel
        inp.seekg(file_header.offset_data, inp.beg);
        // Prepare heading output
        ouput_prepare(inp);
        // Prepare data vector for load data
        data.resize(bmp_info_header.width * bmp_info_header.height * bmp_info_header.bit_count / 8);
        // Row Padding and load data
        row_padding(inp, data);

        // Dynamic Memory Allocation for SIMD
        uint8_t ch    = bmp_info_header.bit_count / 8;
        int simd_size = bmp_info_header.width * bmp_info_header.height / 16 / ch;
        simd_uint8_t aux;
        //data_simd_b = (simd_uint8_t*)_aligned_malloc(simd_size * sizeof(simd_uint8_t), 32);
        //data_simd_g = (simd_uint8_t*)_aligned_malloc(simd_size * sizeof(simd_uint8_t), 32);
        //data_simd_r = (simd_uint8_t*)_aligned_malloc(simd_size * sizeof(simd_uint8_t), 32);

        for(int i = 0; i < data.size(); i += 16*ch){
            aux.a_simd =_mm256_setr_epi16((short)data[i],         (short)data[i + ch],     (short)data[i + 2 * ch], (short)data[i + 3 * ch],
                                          (short)data[i + 4 * ch], (short)data[i + 5 * ch], (short)data[i + 6 * ch], (short)data[i + 7 * ch],
                                          (short)data[i + 8 * ch], (short)data[i + 9 * ch], (short)data[i + 10 * ch],(short)data[i + 11 * ch],
                                          (short)data[i + 12 * ch],(short)data[i + 13 * ch],(short)data[i + 14 * ch],(short)data[i + 15 * ch]);
            
            data_simd_b.push_back(aux);


            aux.a_simd =_mm256_setr_epi16((short)data[i + 1],         (short)data[i + 1 + ch],     (short)data[i + 1 + 2 * ch], (short)data[i + 1 + 3 * ch],
                                          (short)data[i + 1 + 4 * ch], (short)data[i + 1 + 5 * ch], (short)data[i + 1 + 6 * ch], (short)data[i + 1 + 7 * ch],
                                          (short)data[i + 1 + 8 * ch], (short)data[i + 1 + 9 * ch], (short)data[i + 1 + 10 * ch],(short)data[i + 1 + 11 * ch],
                                          (short)data[i + 1 + 12 * ch],(short)data[i + 1 + 13 * ch],(short)data[i + 1 + 14 * ch],(short)data[i + 1 + 15 * ch]);
            data_simd_g.push_back(aux);


            aux.a_simd =_mm256_setr_epi16((short)data[i + 2],         (short)data[i + 2 + ch],     (short)data[i + 2 + 2 * ch], (short)data[i + 2 + 3 * ch],
                                          (short)data[i + 2 + 4 * ch], (short)data[i + 2 + 5 * ch], (short)data[i + 2 + 6 * ch], (short)data[i + 2 + 7 * ch],
                                          (short)data[i + 2 + 8 * ch], (short)data[i + 2 + 9 * ch], (short)data[i + 2 + 10 * ch],(short)data[i + 2 + 11 * ch],
                                          (short)data[i + 2 + 12 * ch],(short)data[i + 2 + 13 * ch],(short)data[i + 2 + 14 * ch],(short)data[i + 2 + 15 * ch]);
            data_simd_r.push_back(aux);
        }
        cout << endl << endl << data_simd_r.size() << endl << endl<< ch<<endl<<endl;
    }

    void write(const char* fname) 
    {
        int ch = bmp_info_header.bit_count / 8;
        
        for (int i = 0, k = 0; i < data.size(); i += 16 * ch, k++) {
            for (int j = 0; j < 15; j++){
                data[i + j * ch]     = uint8_t(data_simd_b[k].a[j]);
                data[i + j * ch + 1] = uint8_t(data_simd_g[k].a[j]);
                data[i + j * ch + 2] = uint8_t(data_simd_r[k].a[j]);
            }
        }
        
        ofstream of{ fname, ios_base::binary };
        if (of) {
            if (bmp_info_header.bit_count == 32) 
            {
                write_headers_and_data(of);
            }
            else if (bmp_info_header.bit_count == 24) 
            {
                if (bmp_info_header.width % 4 == 0) 
                {
                    write_headers_and_data(of);
                }
                else {
                    uint32_t new_stride = row_stride;
                    vector<uint8_t> padding_row(new_stride - row_stride);

                    write_headers(of);

                    for (int y = 0; y < bmp_info_header.height; ++y) 
                    {
                        of.write((const char*)(data.data() + row_stride * y), row_stride);
                        of.write((const char*)padding_row.data(), padding_row.size());
                    }
                }
            }
            else 
            {
                throw runtime_error("Programul nu poate deschide si salva imagini BMP 24 sau BMP 32");
            }
        }
        else
            throw runtime_error("Imaginea nu a putut fi salvata");
    }
    
    void adjust_image(const char* fname)
    {
        for (int j = 0; j < data_simd_b.size(); j++){
            for (int i = 0; i < 16; i++){
                data_simd_b[j].a[i] = (data_simd_b[j].a[i] < 128 && data_simd_b[j].a[i] > 5) ? (data_simd_b[j].a[i] - 5) : 
                                                                  (data_simd_b[j].a[i] < 250)? (data_simd_b[j].a[i] + 5) : 
                                                                                                data_simd_b[j].a[i];

                data_simd_g[j].a[i] = (data_simd_g[j].a[i] < 128 && data_simd_g[j].a[i] > 5)  ? (data_simd_g[j].a[i] - 5) :
                                                                   (data_simd_g[j].a[i] < 250)? (data_simd_g[j].a[i] + 5) :
                                                                                                 data_simd_b[j].a[i];

                data_simd_r[j].a[i] = (data_simd_r[j].a[i] < 128 && data_simd_r[j].a[i] > 5)  ? (data_simd_r[j].a[i] - 5) :
                                                                   (data_simd_r[j].a[i] < 250)? (data_simd_r[j].a[i] + 5) :
                                                                                                 data_simd_r[j].a[i];
            }
        }
    }
    
    void apply_Wavelet(const char* fname)
    {
        int h = bmp_info_header.height;
        int w = bmp_info_header.width;

        simd_uint8_t aux;
        vector<simd_uint8_t>aux_simd_b(data_simd_b.size());
        vector<simd_uint8_t>aux_simd_g(data_simd_g.size());
        vector<simd_uint8_t>aux_simd_r(data_simd_r.size());
        aux_simd_b = data_simd_b;
        aux_simd_g = data_simd_g;
        aux_simd_r = data_simd_r;

        aux.a_simd = _mm256_setr_epi16(2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2);

        for (int i = 0; i < h; i++){
            for (int j = 0; j < w / 2; j++) {
                data_simd_b[w * i + j].a_simd = _mm256_div_epi16(_mm256_adds_epi16(aux_simd_b[w * i + 2 * j].a_simd, aux_simd_b[w * i + 2 * j + 1].a_simd), aux.a_simd);
                data_simd_g[w * i + j].a_simd = _mm256_div_epi16(_mm256_adds_epi16(aux_simd_g[w * i + 2 * j].a_simd, aux_simd_g[w * i + 2 * j + 1].a_simd), aux.a_simd);
                data_simd_r[w * i + j].a_simd = _mm256_div_epi16(_mm256_adds_epi16(aux_simd_r[w * i + 2 * j].a_simd, aux_simd_r[w * i + 2 * j + 1].a_simd), aux.a_simd);
               
                data_simd_b[w * i + j + w / 2].a_simd = _mm256_div_epi16(_mm256_adds_epi16(aux_simd_b[w * i + 2 * j].a_simd, aux_simd_b[w * i + 2 * j + 1].a_simd),aux.a_simd);
                data_simd_g[w * i + j + w / 2].a_simd = _mm256_div_epi16(_mm256_adds_epi16(aux_simd_g[w * i + 2 * j].a_simd, aux_simd_g[w * i + 2 * j + 1].a_simd),aux.a_simd);
                data_simd_r[w * i + j + w / 2].a_simd = _mm256_div_epi16(_mm256_adds_epi16(aux_simd_r[w * i + 2 * j].a_simd, aux_simd_r[w * i + 2 * j + 1].a_simd),aux.a_simd);
            }
        }
        /*
        aux_simd_b = data_simd_b;
        aux_simd_g = data_simd_g;
        aux_simd_r = data_simd_r;

            for (int j = 0; j < w / 2/16; j+=16) 
            {
                for (int i = 0; i < h / 2/16; i+=16) 
                {
                    data_simd_b[h * i + j].a_simd = _mm256_div_epi16(_mm256_adds_epi16(aux_simd_b[2 * h * i + j].a_simd, aux_simd_b[2 * h * i + w + j].a_simd), aux.a_simd);
                    data_simd_g[h * i + j].a_simd = _mm256_div_epi16(_mm256_adds_epi16(aux_simd_g[2 * h * i + j].a_simd, aux_simd_g[2 * h * i + w + j].a_simd), aux.a_simd);
                    data_simd_r[h * i + j].a_simd = _mm256_div_epi16(_mm256_adds_epi16(aux_simd_r[2 * h * i + j].a_simd, aux_simd_r[2 * h * i + w + j].a_simd), aux.a_simd);

                    data_simd_b[h * (i + h / 2) + j].a_simd = _mm256_div_epi16(_mm256_adds_epi16(aux_simd_b[2 * h * i + j].a_simd, aux_simd_b[2 * h * i + w + j].a_simd), aux.a_simd);
                    data_simd_g[h * (i + h / 2) + j].a_simd = _mm256_div_epi16(_mm256_adds_epi16(aux_simd_g[2 * h * i + j].a_simd, aux_simd_g[2 * h * i + w + j].a_simd), aux.a_simd);
                    data_simd_r[h * (i + h / 2) + j].a_simd = _mm256_div_epi16(_mm256_adds_epi16(aux_simd_r[2 * h * i + j].a_simd, aux_simd_r[2 * h * i + w + j].a_simd), aux.a_simd);
                }
            }  */
    }
    
    /*
    void hide(const char* filename2)
    {
        BMP bmp2;
        bmp2.readBMP(filename2);

        
        vector<u_int> arr_hid;
        u_int aux;
        
        uint8_t channels = bmp2.bmp_info_header.bit_count / 8;
        
        for (uint32_t y = 0; y < bmp2.bmp_info_header.height; y++) 
        {
            for (uint32_t x = 0; x < bmp2.bmp_info_header.width; x++) 
            {
                u_int aux;
                aux.a_simd = _mm_set_epi32(((int)data[channels * (y * bmp2.bmp_info_header.width + x)] & 0xc0)     >> 6,
                                         ((int)data[channels * (y * bmp2.bmp_info_header.width + x)] & 0xc0 + 1) >> 6,
                                         ((int)data[channels * (y * bmp2.bmp_info_header.width + x)] & 0xc0 + 2) >> 6,
                                          0);
                arr_hid.push_back(aux);

                aux.a_simd = _mm_set_epi32(((int)data[channels * (y * bmp2.bmp_info_header.width + x)] & 0x30)     >> 4,
                                         ((int)data[channels * (y * bmp2.bmp_info_header.width + x)] & 0x30 + 1) >> 4,
                                         ((int)data[channels * (y * bmp2.bmp_info_header.width + x)] & 0x30 + 2) >> 4,
                                         0);
                arr_hid.push_back(aux);

                aux.a_simd = _mm_set_epi32(((int)data[channels * (y * bmp2.bmp_info_header.width + x)] & 0x0c)     >> 2,
                                         ((int)data[channels * (y * bmp2.bmp_info_header.width + x)] & 0x0c + 1) >> 2,
                                         ((int)data[channels * (y * bmp2.bmp_info_header.width + x)] & 0x0c + 2) >> 2,
                                         0);
                arr_hid.push_back(aux);

                aux.a_simd = _mm_set_epi32(((int)data[channels * (y * bmp2.bmp_info_header.width + x)] & 0x03),
                                         ((int)data[channels * (y * bmp2.bmp_info_header.width + x)] & 0x03),
                                         ((int)data[channels * (y * bmp2.bmp_info_header.width + x)] & 0x03),
                                         0);
                arr_hid.push_back(aux);
            }
        }
        
        aux.a_simd = _mm_setr_epi32(0xfffffc,0xfffffc,0xfffffc,0xfffffc);

        vector<u_int> data_simd_int(data_simd.size());
        
        for (int i = 0; i < data_simd.size(); i++)
        {
            data_simd_int[i].a[0] = (int)data_simd[i].a[0];
            data_simd_int[i].a[1] = (int)data_simd[i].a[1];
            data_simd_int[i].a[2] = (int)data_simd[i].a[2];
            data_simd_int[i].a[3] = (int)data_simd[i].a[3];
        }
        
        for (int i = 0; i < arr_hid.size(); i++)
        {
            data_simd_int[i].a_simd = _mm_xor_si128(data_simd_int[i].a_simd, aux.a_simd);
            data_simd_int[i].a_simd = _mm_or_si128(data_simd_int[i].a_simd, arr_hid[i].a_simd);
        }
        
        //tranformare inversa pe linii
        for (int j = 0; j < bmp_info_header.width / 2; j++)
        {
            for (int i = 0; i < bmp_info_header.height / 2; i++)
            {
                data_simd_int[bmp_info_header.height * (2 * i) + j].a_simd     = _mm_add_epi32(data_simd_int[bmp_info_header.height * i + j].a_simd, data_simd_int[bmp_info_header.height * (i + bmp_info_header.width/2) + j].a_simd);
                data_simd_int[bmp_info_header.height * (2 * i + 1) + j].a_simd = _mm_sub_epi32(data_simd_int[bmp_info_header.height * i + j].a_simd, data_simd_int[bmp_info_header.height * (i + bmp_info_header.width/2) + j].a_simd);
            }
        }
        
        for (int i = 0; i < data_simd.size(); i++)
        {
            data_simd[i].a[0] = (float)data_simd_int[i].a[0];
            data_simd[i].a[1] = (float)data_simd_int[i].a[1];
            data_simd[i].a[2] = (float)data_simd_int[i].a[2];
            data_simd[i].a[3] = (float)data_simd_int[i].a[3];
        }
        
    }*/

    private:
        uint32_t row_stride{ 0 };

        void check(ifstream& inp)
        {
            if (inp)
            {
                inp.read((char*)&file_header, sizeof(file_header));
                //Check if image is BMP
                if (file_header.file_type != 0x4D42)
                {
                    cout << "ERROR! This is not a BMP image! T-T";
                }
                inp.read((char*)&bmp_info_header, sizeof(bmp_info_header));

                if (bmp_info_header.bit_count == 32) {
                    // Check if image has info about RGB mask
                    if (bmp_info_header.size >= (sizeof(BMPInfoHeader) + sizeof(BMPColorHeader))) {
                        inp.read((char*)&bmp_color_header, sizeof(bmp_color_header));
                        // Check if RGB space is RGBA
                        BMPColorHeader expected_color_header;
                        if (expected_color_header.red_mask != bmp_color_header.red_mask ||
                            expected_color_header.blue_mask != bmp_color_header.blue_mask ||
                            expected_color_header.green_mask != bmp_color_header.green_mask ||
                            expected_color_header.alpha_mask != bmp_color_header.alpha_mask) {
                            cout << "ERROR! RGB space is not RGBA! T-T";
                        }
                    }
                    else {
                        cout << "ERROR! Image has no info about RGB! T-T";
                    }
                }
            }
            else
                throw runtime_error("Image could not be open! (T-T) ");

        }

        void ouput_prepare(ifstream& inp)
        {
            // Make output header
            if (bmp_info_header.bit_count == 32)
            {
                bmp_info_header.size = sizeof(BMPInfoHeader) + sizeof(BMPColorHeader);
                file_header.offset_data = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + sizeof(BMPColorHeader);
            }
            else
            {
                bmp_info_header.size = sizeof(BMPInfoHeader);
                file_header.offset_data = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);
            }
            file_header.file_size = file_header.offset_data;
        }

        void write_headers(ofstream& of)
        {
            of.write((const char*)&file_header, sizeof(file_header));
            of.write((const char*)&bmp_info_header, sizeof(bmp_info_header));
            if (bmp_info_header.bit_count == 32)
                of.write((const char*)&bmp_color_header, sizeof(bmp_color_header));

        }

        void write_headers_and_data(ofstream& of)
        {
            write_headers(of);
            of.write((const char*)data.data(), data.size());
        }

        void row_padding(ifstream& inp, vector<uint8_t>&data) {
            if (bmp_info_header.width % 4 == 0)
            {
                inp.read((char*)data.data(), data.size());
                file_header.file_size += static_cast<uint32_t>(data.size());
            }
            else
            {
                row_stride = bmp_info_header.width * bmp_info_header.bit_count / 8;
                uint32_t new_stride = row_stride;
                vector<uint8_t> padding_row(new_stride - row_stride);

                for (int y = 0; y < bmp_info_header.height; ++y)
                {
                    inp.read((char*)(data.data() + row_stride * y), row_stride);
                    inp.read((char*)padding_row.data(), padding_row.size());
                }
                file_header.file_size += static_cast<uint32_t>(data.size()) + bmp_info_header.height * static_cast<uint32_t>(padding_row.size());
            }
        }
};


int main()
{
    BMP bmp;

    bmp.readBMP("flower1.bmp");
    bmp.adjust_image("flower1.bmp");
    bmp.apply_Wavelet("flower1.bmp");
    //bmp.hide("lena.bmp");
    bmp.write  ("flower2.bmp");
    

    //bmp2.readBMP("lena.bmp");
    //bmp2.write("lena2.bmp", bmp2.data, bmp2.data_simd);
	return 0;
}

