#include <iostream>
#include <vector>
#include <fstream>
#include <inttypes.h>
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

	//Image dimensions
	int h;  //Input image height
	int w;  //Input image weight
	int ch; //Input image number of channels

	//Image data
	vector<uint8_t> data;

	//SIMD data
	union simd_union
	{
		__m128 a_simd;
		float  a[4];
	};

	vector<simd_union> data_simd;
	vector<simd_union> aux_simd;
	simd_union aux;

	void readBMP(const char* filename)
	{
		//Read BMP file
		ifstream inp{ filename, ios_base::binary };

		//Check if BMP file exist
		check(inp);

		//Look for start pixel
		inp.seekg(file_header.offset_data, inp.beg);

		//Make output header
		prepare_output_header(inp);

		//Get image dimensions
		ch = bmp_info_header.bit_count / 8;
		h = bmp_info_header.height;
		w = bmp_info_header.width;

		//Prepare vector for load data
		data.resize(w * h * ch);

		//Row padding and load data
		load_data(inp);

		//Load Data in SIMD vector
		for (int i = 0; i < h * w; i++) {
			aux.a_simd = _mm_setr_ps((float)data[ch * i], (float)data[ch * i + 1], (float)data[ch * i + 2], 0);
			data_simd.push_back(aux);
		}

	}

	void writeBMP(const char* fname)
	{
		int j = 0;
		for (int i = 0; i < h * w; i++) {
			data[ch * i] = (uint8_t)data_simd[j].a[0];
			data[ch * i + 1] = (uint8_t)data_simd[j].a[1];
			data[ch * i + 2] = (uint8_t)data_simd[j].a[2];
			j++;
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
				throw runtime_error("! (T-T) ! ERROR ! Programul nu poate deschide si salva imagini BMP 24 sau BMP 32");
			}
		}
		else
			throw runtime_error("! (T-T) ! ERROR ! Imaginea nu a putut fi salvata");
	}

	void adjust_image()
	{
		for (uint32_t i = 0; i < data_simd.size(); i++)
		{
			data_simd[i].a[0] = (data_simd[i].a[0] < 128 && data_simd[i].a[0] > 5) ? (data_simd[i].a[0] - 5) : (data_simd[i].a[0] < 250) ? (data_simd[i].a[0] + 5) : data_simd[i].a[0];
			data_simd[i].a[1] = (data_simd[i].a[1] < 128 && data_simd[i].a[1] > 5) ? (data_simd[i].a[1] - 5) : (data_simd[i].a[1] < 250) ? (data_simd[i].a[1] + 5) : data_simd[i].a[1];
			data_simd[i].a[2] = (data_simd[i].a[2] < 128 && data_simd[i].a[2] > 5) ? (data_simd[i].a[2] - 5) : (data_simd[i].a[2] < 250) ? (data_simd[i].a[2] + 5) : data_simd[i].a[2];
		}
	}

	void apply_Wavelet()
	{
		aux.a_simd = _mm_setr_ps(2, 2, 2, 2);
		aux_simd = data_simd;
		for (int i = 0; i < h; i++)
		{
			for (int j = 0; j < w / 2; j++)
			{
				data_simd[w * i + j].a_simd = _mm_div_ps(_mm_add_ps(aux_simd[w * i + 2 * j].a_simd, aux_simd[w * i + 2 * j + 1].a_simd), aux.a_simd);
				data_simd[w * i + j + w / 2].a_simd = _mm_div_ps(_mm_sub_ps(aux_simd[w * i + 2 * j].a_simd, aux_simd[w * i + 2 * j + 1].a_simd), aux.a_simd);
			}
		}

		aux_simd = data_simd;

		for (int j = 0; j < w / 2; j++)
		{
			for (int i = 0; i < h / 2; i++)
			{
				data_simd[h * i + j].a_simd = _mm_div_ps(_mm_sub_ps(aux_simd[2 * h * i + j].a_simd, aux_simd[2 * h * i + w + j].a_simd), aux.a_simd);
				data_simd[h * (i + h / 2) + j].a_simd = _mm_div_ps(_mm_add_ps(aux_simd[2 * h * i + j].a_simd, aux_simd[2 * h * i + w + j].a_simd), aux.a_simd);
			}
		}
	}

	void hide(const char* filename2)
	{
		BMP bmp2;
		bmp2.readBMP(filename2);

		vector<simd_union>arr_hid;

		int bmp2_h = bmp2.bmp_info_header.height;
		int bmp2_w = bmp2.bmp_info_header.width;

		for (int y = 0; y < bmp2_h; y++)
		{
			for (int x = 0; x < bmp2_w; x++)
			{
				aux.a_simd = _mm_setr_ps((int)((data[(bmp2_w * y) + (4 * x)] & 0xc0) >> 6),
					(int)((data[(bmp2_w * y) + (4 * x) + 1] & 0xc0) >> 6),
					(int)((data[(bmp2_w * y) + (4 * x) + 2] & 0xc0) >> 6),
					0);
				arr_hid.push_back(aux);

				aux.a_simd = _mm_setr_ps((int)((data[(bmp2_w * y) + (4 * x)] & 0x30) >> 4),
					(int)((data[(bmp2_w * y) + (4 * x) + 1] & 0x30) >> 4),
					(int)((data[(bmp2_w * y) + (4 * x) + 2] & 0x30) >> 4),
					0);
				arr_hid.push_back(aux);

				aux.a_simd = _mm_setr_ps(((int)data[(bmp2_w * y) + (4 * x)] & 0x0c) >> 2,
					((int)data[(bmp2_w * y) + (4 * x) + 1] & 0x0c) >> 2,
					((int)data[(bmp2_w * y) + (4 * x) + 2] & 0x0c) >> 2,
					0);
				arr_hid.push_back(aux);

				aux.a_simd = _mm_setr_ps(((int)data[(bmp2_w * y) + (4 * x)] & 0x03),
					((int)data[(bmp2_w * y) + (4 * x) + 1] & 0x03),
					((int)data[(bmp2_w * y) + (4 * x) + 2] & 0x03),
					0);
				arr_hid.push_back(aux);
			}
		}

		aux.a_simd = _mm_setr_ps(0xfffffc, 0xfffffc, 0xfffffc, 0xfffffc);

		for (int y = 0; y < bmp2_h; y++) {
			for (int x = 0; x < bmp2_w; x++) {
				data_simd[(bmp2_w * y) + (4 * x)].a_simd = _mm_xor_ps(data_simd[(bmp2_w * y) + (4 * x)].a_simd, aux.a_simd);
				data_simd[(bmp2_w * y) + (4 * x)].a_simd = _mm_or_ps(data_simd[(bmp2_w * y) + (4 * x)].a_simd, arr_hid[(bmp2_w * y) + (4 * x)].a_simd);
			}
		}

		//tranformare inversa pe linii
		for (int j = 0; j < w / 2; j++) {
			for (int i = 0; i < h / 2; i++) {
				aux_simd[h * (2 * i) + j].a_simd = _mm_add_ps(data_simd[h * i + j].a_simd, data_simd[h * (i + w / 2) + j].a_simd);
				aux_simd[h * (2 * i + 1) + j].a_simd = _mm_sub_ps(data_simd[h * (i + w / 2) + j].a_simd, data_simd[h * i + j].a_simd);
			}
		}

		//tranformare inversa pe coloana
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w / 2; j++) {
				data_simd[w * i + 2 * j].a_simd = _mm_add_ps(aux_simd[w * i + j].a_simd, aux_simd[w * i + j + h / 2].a_simd);
				data_simd[w * i + 2 * j + 1].a_simd = _mm_sub_ps(aux_simd[w * i + j].a_simd, aux_simd[w * i + j + h / 2].a_simd);
			}
		}

	}



private:
	uint32_t row_stride{ 0 };

	void check(ifstream& inp) {
		if (inp)
		{
			inp.read((char*)&file_header, sizeof(file_header));
			//Check if image is BMP
			if (file_header.file_type != 0x4D42)
			{
				throw runtime_error("! (T-T) ! ERROR ! This is not a BMP image!");
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
						throw runtime_error("! (T-T) ! ERROR ! RGB space is not RGBA!");
					}
				}
				else {
					throw runtime_error("! (T-T) ! ERROR ! Image has no info about RGB!");
				}
			}
		}
		else
			throw runtime_error("! (T-T) ! ERROR ! Image could not be open!");

	}

	void prepare_output_header(ifstream& inp) {
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

	void load_data(ifstream& inp)
	{
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
};




int main()
{
	BMP bmp;
	bmp.readBMP("flower1.bmp");
	bmp.adjust_image();
	bmp.apply_Wavelet();
	bmp.hide("lena.bmp");
	bmp.writeBMP("flower2.bmp");

	return 0;
}

