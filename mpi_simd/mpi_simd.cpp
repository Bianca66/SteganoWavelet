#include <iostream>
#include <vector>
#include <fstream>
#include <inttypes.h>
#include <immintrin.h>
#include <mpi.h>
#define MASTER 0

using namespace std;

MPI_Datatype MPI_simd;

int total_proc;
int mpi_rank;

float* a;
float* aux;
float* hd;

__m128* data_simd;
__m128* aux_simd;
__m128* hide_simd;

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


	void readBMP(const char* filename)
	{
		//Read BMP file
		ifstream inp{ filename, ios_base::binary };
		//MPI_File inp;
		//MPI_Status status;
		//int inp_exist = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDWR, MPI_INFO_NULL, &inp);
		//MPI_File_read(inp, &data, data.size(), MPI_UINT8_T, &status);
		//Check if BMP file exist
		//check(inp, inp_exist);
		check(inp);
		//Look for start pixel
		//	MPI_Offset a = file_header.offset_data;
		//	int beg = MPI_File_get_position_shared(inp, &a);
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
		a = (float*)_mm_malloc(sizeof(float) * w * h * 4, 32); // mm malloc aligns to 32 bytes
		aux = (float*)_mm_malloc(sizeof(float) * w * h * 4, 32); // mm malloc aligns to 32 bytes

		data_simd = (__m128*) a;
		aux_simd = (__m128*) aux;

		long long int n = w * h;
		long long int n_per_proc = n / total_proc;

		float* ap;
		float* datap;

		__m128* datap_simd;

		if (mpi_rank == MASTER) //you choose process rank 0 to be your root which will be used to perform input output. 
		{
			// you get the total number of test cases
			MPI_Bcast(&n, 1, MPI_LONG_LONG_INT, MASTER, MPI_COMM_WORLD);
			n_per_proc = n / total_proc;

			ap = (float*)_mm_malloc(sizeof(float) * n_per_proc * 4, 32);
			datap = (float*)_mm_malloc(sizeof(float) * n_per_proc * 4, 32);

			datap_simd = (__m128*) ap;

			//Broadcast element per process
			MPI_Bcast(&n_per_proc, 1, MPI_LONG_LONG_INT, MASTER, MPI_COMM_WORLD);
			//scattering array data  
			MPI_Scatter(&data, n_per_proc, MPI_UINT8_T, datap, n_per_proc, MPI_UINT8_T, 0, MPI_COMM_WORLD);
			for (int i = 0; i < n_per_proc; i++)
				datap_simd[i] = _mm_setr_ps((float)data[ch * i], (float)data[ch * i + 1], (float)data[ch * i + 2], 0);
			MPI_Gather(datap_simd, n_per_proc, MPI_simd, data_simd, n_per_proc, MPI_simd, MASTER, MPI_COMM_WORLD);
		}
		//MPI_Barrier(MPI_COMM_WORLD);
	}

	void writeBMP(const char* fname)
	{

		for (int i = 0, j = 0; i < h * w; i++, j += 4)
		{
			data[ch * i] = a[j];
			data[ch * i + 1] = a[j + 1];
			data[ch * i + 2] = a[j + 2];
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
		_mm_free(a);
		_mm_free(aux);
	}

	void adjust_image()
	{
		float* ap;
		float* auxp;

		__m128* ap_simd;
		__m128* auxp_simd;

		long long int n, n_per_proc;

		if (mpi_rank == MASTER)
		{
			n = w * h * ch;
			MPI_Bcast(&n, 1, MPI_LONG_LONG_INT, MASTER, MPI_COMM_WORLD);
			n_per_proc = n / total_proc;

			ap = (float*)_mm_malloc(sizeof(float) * n_per_proc * 4, 32);
			auxp = (float*)_mm_malloc(sizeof(float) * n_per_proc * 4, 32);

			ap_simd = (__m128*) ap;
			auxp_simd = (__m128*) auxp;

			MPI_Bcast(&n_per_proc, 1, MPI_LONG_LONG_INT, MASTER, MPI_COMM_WORLD);
			MPI_Scatter(a, n_per_proc, MPI_FLOAT, ap, n_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);
			for (int i = 0; i < n_per_proc; i++)
				auxp[i] = (a[i] < 128 && a[i] > 5) ? (a[i] - 5) : (a[i] > 128 && a[i] < 250) ? (a[i] + 5) : a[i];
			MPI_Gather(auxp, n_per_proc, MPI_FLOAT, a, n_per_proc, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

			_mm_free(ap);
			_mm_free(auxp);
		}
	}

	void apply_Wavelet()
	{
		__m128 aux_aux;
		long long int n;
		long long int n_per_proc;

		float* ap;
		float* auxp;

		__m128* ap_simd;
		__m128* auxp_simd;


		if (mpi_rank == MASTER)
		{
			n = w * h;
			n_per_proc = n / total_proc;

			MPI_Bcast(&n, 1, MPI_LONG_LONG_INT, MASTER, MPI_COMM_WORLD);
			MPI_Bcast(&n_per_proc, 1, MPI_LONG_LONG_INT, MASTER, MPI_COMM_WORLD);

			ap = (float*)_mm_malloc(sizeof(float) * n_per_proc * 4, 32);
			auxp = (float*)_mm_malloc(sizeof(float) * n_per_proc * 4, 32);

			ap_simd = (__m128*) ap;
			auxp_simd = (__m128*) auxp;

			MPI_Scatter(data_simd, n_per_proc, MPI_simd, ap_simd, n_per_proc, MPI_simd, 0, MPI_COMM_WORLD);
			for (int i = 0; i < n_per_proc; i++)
				auxp_simd[i] = ap_simd[i];
			MPI_Gather(auxp_simd, n_per_proc, MPI_simd, aux_simd, n_per_proc, MPI_simd, MASTER, MPI_COMM_WORLD);


			aux_aux = _mm_setr_ps(2, 2, 2, 2);

			//****************************************************************
			n = h;
			n_per_proc = n / total_proc;

			MPI_Bcast(&n, 1, MPI_LONG_LONG_INT, MASTER, MPI_COMM_WORLD);
			MPI_Bcast(&n_per_proc, 1, MPI_LONG_LONG_INT, MASTER, MPI_COMM_WORLD);
			MPI_Bcast(&aux_aux, 1, MPI_simd, MASTER, MPI_COMM_WORLD);

			ap = (float*)_mm_malloc(sizeof(float) * n_per_proc * 4, 32);
			auxp = (float*)_mm_malloc(sizeof(float) * n_per_proc * 4, 32);

			ap_simd = (__m128*) ap;
			auxp_simd = (__m128*) auxp;

			MPI_Scatter(aux_simd, n_per_proc, MPI_simd, auxp_simd, n_per_proc, MPI_simd, 0, MPI_COMM_WORLD);

			for (int i = 0; i < n_per_proc; i++)
			{
				for (int j = 0; j < w / 2; j++)
				{
					ap_simd[w * i + j] = _mm_div_ps(_mm_add_ps(auxp_simd[w * i + 2 * j], auxp_simd[w * i + 2 * j + 1]), aux_aux);
					ap_simd[w * i + j + w / 2] = _mm_div_ps(_mm_sub_ps(auxp_simd[w * i + 2 * j], auxp_simd[w * i + 2 * j + 1]), aux_aux);
				}
			}
			MPI_Gather(ap_simd, n_per_proc, MPI_simd, data_simd, n_per_proc, MPI_simd, MASTER, MPI_COMM_WORLD);

			//****************************************************************

			n = w * h;
			n_per_proc = n / total_proc;

			MPI_Bcast(&n, 1, MPI_LONG_LONG_INT, MASTER, MPI_COMM_WORLD);
			MPI_Bcast(&n_per_proc, 1, MPI_LONG_LONG_INT, MASTER, MPI_COMM_WORLD);

			ap = (float*)_mm_malloc(sizeof(float) * n_per_proc * 4, 32);
			auxp = (float*)_mm_malloc(sizeof(float) * n_per_proc * 4, 32);

			ap_simd = (__m128*) ap;
			auxp_simd = (__m128*) auxp;

			MPI_Scatter(data_simd, n_per_proc, MPI_simd, ap_simd, n_per_proc, MPI_simd, 0, MPI_COMM_WORLD);
			for (int i = 0; i < n_per_proc; i++)
				auxp_simd[i] = ap_simd[i];
			MPI_Gather(auxp_simd, n_per_proc, MPI_simd, aux_simd, n_per_proc, MPI_simd, MASTER, MPI_COMM_WORLD);

			//****************************************************************

			n = w / 2;
			n_per_proc = n / total_proc;

			MPI_Bcast(&n, 1, MPI_LONG_LONG_INT, MASTER, MPI_COMM_WORLD);
			MPI_Bcast(&n_per_proc, 1, MPI_LONG_LONG_INT, MASTER, MPI_COMM_WORLD);
			MPI_Bcast(&aux_aux, 1, MPI_simd, MASTER, MPI_COMM_WORLD);

			ap = (float*)_mm_malloc(sizeof(float) * n_per_proc * 4, 32);
			auxp = (float*)_mm_malloc(sizeof(float) * n_per_proc * 4, 32);

			ap_simd = (__m128*) ap;
			auxp_simd = (__m128*) auxp;

			MPI_Scatter(aux_simd, n_per_proc, MPI_simd, auxp_simd, n_per_proc, MPI_simd, 0, MPI_COMM_WORLD);

			for (int j = 0; j < n_per_proc; j++)
			{
				for (int i = 0; i < h / 2; i++)
				{
					ap_simd[h * i + j] = _mm_div_ps(_mm_sub_ps(auxp_simd[2 * h * i + j], auxp_simd[2 * h * i + w + j]), aux_aux);
					ap_simd[h * (i + h / 2) + j] = _mm_div_ps(_mm_add_ps(auxp_simd[2 * h * i + j], auxp_simd[2 * h * i + w + j]), aux_aux);
				}
			}
			MPI_Gather(ap_simd, n_per_proc, MPI_simd, data_simd, n_per_proc, MPI_simd, MASTER, MPI_COMM_WORLD);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

	void hide(const char* filename2)
	{
		//****************************************************************
		BMPFileHeader  file_header_hiden;
		BMPInfoHeader  bmp_info_header_hiden;
		BMPColorHeader bmp_color_header_hiden;

		//Read BMP hiden file
		ifstream inp2{ filename2, ios_base::binary };

		inp2.read((char*)&file_header_hiden, sizeof(file_header_hiden));
		inp2.seekg(file_header_hiden.offset_data, inp2.beg);

		//Get image dimensions
		int inp2_ch = bmp_info_header_hiden.bit_count / 8;
		int inp2_h = bmp_info_header_hiden.height;
		int inp2_w = bmp_info_header_hiden.width;

		vector<uint8_t> data2;
		//Prepare vector for load data
		data2.resize(inp2_w * inp2_h * inp2_ch);

		//Row padding and load data
		if (bmp_info_header_hiden.width % 4 == 0)
		{
			inp2.read((char*)data2.data(), data2.size());
			file_header_hiden.file_size += static_cast<uint32_t>(data2.size());
		}
		else
		{
			uint32_t row_stride2{ 0 };
			row_stride2 = inp2_w * inp2_ch;
			uint32_t new_stride2 = row_stride2;
			vector<uint8_t> padding_row2(new_stride2 - row_stride2);

			for (int y = 0; y < bmp_info_header.height; ++y)
			{
				inp2.read((char*)(data2.data() + row_stride2 * y), row_stride2);
				inp2.read((char*)padding_row2.data(), padding_row2.size());
			}
			file_header.file_size += static_cast<uint32_t>(data2.size()) + bmp_info_header.height * static_cast<uint32_t>(padding_row2.size());
		}
		//****************************************************************

		float* ap;
		float* auxp;
		float* datap;
		uint8_t* data2p;

		__m128* ap_simd;
		__m128* auxp_simd;
		__m128* datap_simd;

		long long int n;
		long long int n_per_proc;

		hd = (float*)_mm_malloc(sizeof(float) * inp2_h * inp2_w * 4, 32);
		data2p = (uint8_t*)_mm_malloc(sizeof(uint8_t) * inp2_h * inp2_w * 4, 8);
		hide_simd = (__m128*) hd;

		if (mpi_rank == MASTER)
		{
			n = inp2_h;
			n_per_proc = n / total_proc;

			MPI_Bcast(&n, 1, MPI_LONG_LONG_INT, MASTER, MPI_COMM_WORLD);

			ap = (float*)_mm_malloc(sizeof(float) * n_per_proc * 4, 32);
			auxp = (float*)_mm_malloc(sizeof(float) * n_per_proc * 4, 32);

			ap_simd = (__m128*) ap;
			auxp_simd = (__m128*) auxp;

			MPI_Bcast(&n_per_proc, 1, MPI_LONG_LONG_INT, MASTER, MPI_COMM_WORLD);
			MPI_Scatter(&data2, n_per_proc, MPI_UINT8_T, data2p, n_per_proc, MPI_UINT8_T, 0, MPI_COMM_WORLD);
			for (int y = 0; y < n_per_proc; y++)
			{
				for (int x = 0, i = 0; x < inp2_w; x++, i += 4)
				{
					auxp_simd[i] = _mm_setr_ps((int)(data2p[(inp2_w * y) + (4 * x)] & 0xc0) >> 6,
						(int)(data2p[(inp2_w * y) + (4 * x) + 1] & 0xc0) >> 6,
						(int)(data2p[(inp2_w * y) + (4 * x) + 2] & 0xc0) >> 6,
						0);

					auxp_simd[i + 1] = _mm_setr_ps((int)(data2p[(inp2_w * y) + (4 * x)] & 0x30) >> 4,
						(int)(data2p[(inp2_w * y) + (4 * x) + 1] & 0x30) >> 4,
						(int)(data2p[(inp2_w * y) + (4 * x) + 2] & 0x30) >> 4,
						0);

					auxp_simd[i + 2] = _mm_setr_ps((int)(data2p[(inp2_w * y) + (4 * x)] & 0x0c) >> 2,
						(int)(data2p[(inp2_w * y) + (4 * x) + 1] & 0x0c) >> 2,
						(int)(data2p[(inp2_w * y) + (4 * x) + 2] & 0x0c) >> 2,
						0);

					auxp_simd[i + 3] = _mm_setr_ps((int)(data2p[(inp2_w * y) + (4 * x)] & 0x03),
						(int)(data2p[(inp2_w * y) + (4 * x) + 1] & 0x03),
						(int)(data2p[(inp2_w * y) + (4 * x) + 2] & 0x03),
						0);
				}
			}
			MPI_Gather(auxp_simd, n_per_proc, MPI_simd, hide_simd, n_per_proc, MPI_simd, MASTER, MPI_COMM_WORLD);

			//*********************************************************************************
			__m128 aux_aux = _mm_setr_ps(0xfffffc, 0xfffffc, 0xfffffc, 0xfffffc);

			n = inp2_h;
			n_per_proc = n / total_proc;

			MPI_Bcast(&n, 1, MPI_LONG_LONG_INT, MASTER, MPI_COMM_WORLD);
			MPI_Bcast(&n_per_proc, 1, MPI_LONG_LONG_INT, MASTER, MPI_COMM_WORLD);
			MPI_Bcast(&aux, 1, MPI_simd, MASTER, MPI_COMM_WORLD);

			ap = (float*)_mm_malloc(sizeof(float) * n_per_proc * 4, 32);
			auxp = (float*)_mm_malloc(sizeof(float) * n_per_proc * 4, 32);
			datap = (float*)_mm_malloc(sizeof(float) * n_per_proc * 4, 32);

			ap_simd = (__m128*) ap;
			auxp_simd = (__m128*) auxp;
			datap_simd = (__m128*) datap;

			MPI_Scatter(&data_simd, n_per_proc, MPI_simd, datap_simd, n_per_proc, MPI_simd, 0, MPI_COMM_WORLD);
			MPI_Scatter(&hide_simd, n_per_proc, MPI_simd, auxp_simd, n_per_proc, MPI_simd, 0, MPI_COMM_WORLD);
			for (int y = 0; y < n_per_proc; y++)
			{
				for (int x = 0; x < inp2_w; x++)
				{
					ap_simd[(inp2_w * y) + (4 * x)] = _mm_xor_ps(datap_simd[(inp2_w * y) + (4 * x)], aux_aux);
					ap_simd[(inp2_h * y) + (4 * x)] = _mm_or_ps(datap_simd[(inp2_w * y) + (4 * x)], auxp_simd[(inp2_w * y) + (4 * x)]);
				}
			}
			MPI_Gather(ap_simd, n_per_proc, MPI_simd, data_simd, n_per_proc, MPI_simd, MASTER, MPI_COMM_WORLD);

			//*******************************************************************************

			n = w / 2;
			n_per_proc = n / total_proc;

			MPI_Bcast(&n, 1, MPI_LONG_LONG_INT, MASTER, MPI_COMM_WORLD);
			MPI_Bcast(&n_per_proc, 1, MPI_LONG_LONG_INT, MASTER, MPI_COMM_WORLD);
			MPI_Bcast(&aux, 1, MPI_simd, MASTER, MPI_COMM_WORLD);

			ap = (float*)_mm_malloc(sizeof(float) * n_per_proc * 4, 32);
			auxp = (float*)_mm_malloc(sizeof(float) * n_per_proc * 4, 32);
			datap = (float*)_mm_malloc(sizeof(float) * n_per_proc * 4, 32);

			ap_simd = (__m128*) ap;
			auxp_simd = (__m128*) auxp;
			datap_simd = (__m128*) datap;

			MPI_Scatter(&data_simd, n_per_proc, MPI_simd, datap_simd, n_per_proc, MPI_simd, 0, MPI_COMM_WORLD);
			for (int j = 0; j < n_per_proc; j++)
			{
				for (int i = 0; i < h / 2; i++)
				{
					auxp_simd[h * (2 * i) + j] = _mm_add_ps(datap_simd[h * i + j], datap_simd[h * (i + w / 2) + j]);
					auxp_simd[h * (2 * i + 1) + j] = _mm_sub_ps(datap_simd[h * (i + w / 2) + j], datap_simd[h * i + j]);
				}
			}
			MPI_Gather(auxp_simd, n_per_proc, MPI_simd, aux_simd, n_per_proc, MPI_simd, MASTER, MPI_COMM_WORLD);

			//**************************************************************************
			n = h;
			n_per_proc = n / total_proc;

			MPI_Bcast(&n, 1, MPI_LONG_LONG_INT, MASTER, MPI_COMM_WORLD);
			MPI_Bcast(&n_per_proc, 1, MPI_LONG_LONG_INT, MASTER, MPI_COMM_WORLD);

			auxp = (float*)_mm_malloc(sizeof(float) * n_per_proc * 4, 32);
			datap = (float*)_mm_malloc(sizeof(float) * n_per_proc * 4, 32);

			auxp_simd = (__m128*) auxp;
			datap_simd = (__m128*) datap;

			MPI_Scatter(&aux_simd, n_per_proc, MPI_simd, auxp_simd, n_per_proc, MPI_simd, 0, MPI_COMM_WORLD);
			for (int i = 0; i < n_per_proc; i++)
			{
				for (int j = 0; j < w / 2; j++) {
					datap_simd[w * i + 2 * j] = _mm_add_ps(auxp_simd[w * i + j], auxp_simd[w * i + j + h / 2]);
					datap_simd[w * i + 2 * j + 1] = _mm_sub_ps(auxp_simd[w * i + j], auxp_simd[w * i + j + h / 2]);
				}
			}
			MPI_Gather(datap_simd, n_per_proc, MPI_simd, data_simd, n_per_proc, MPI_simd, MASTER, MPI_COMM_WORLD);

			MPI_Barrier(MPI_COMM_WORLD);
			_mm_free(hd);
			_mm_free(ap);
			_mm_free(auxp);
			_mm_free(datap);
			_mm_free(data2p);
		}
	}


private:
	uint32_t row_stride{ 0 };
	/*
	void check(MPI_File& inp, int& inp_exist) {
	if (!inp_exist)
	{
	MPI_File_read(inp, &file_header, sizeof(file_header), MPI_CHAR, &status);
	//Check if image is BMP
	if (file_header.file_type != 0x4D42)
	{
	throw runtime_error("! (T-T) ! ERROR ! This is not a BMP image!");
	}

	MPI_File_read(inp, &bmp_info_header, sizeof(bmp_info_header), MPI_CHAR, &status);
	if (bmp_info_header.bit_count == 32) {
	// Check if image has info about RGB mask
	if (bmp_info_header.size >= (sizeof(BMPInfoHeader) + sizeof(BMPColorHeader))) {
	MPI_File_read(inp, &bmp_color_header, sizeof(bmp_color_header), MPI_CHAR, &status);
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
	*/

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
	MPI_Init(NULL, NULL);

	MPI_Type_contiguous(4, MPI_CHAR, &MPI_simd);
	MPI_Type_commit(&MPI_simd);
	MPI_Comm_size(MPI_COMM_WORLD, &total_proc);
	//Now you know the total number of processes running in parallel
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	//Now you know the rank of the current process

	BMP bmp;
	bmp.readBMP("flower1.bmp");
	bmp.adjust_image();
	bmp.apply_Wavelet();
	bmp.hide("lena.bmp");
	bmp.writeBMP("flower22.bmp");
	MPI_Finalize();

	return 0;
}

