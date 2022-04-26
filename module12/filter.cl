__kernel void filter(__global int* buffer, int bufferSize) {
    size_t id = get_global_id(0);
    size_t lid = get_local_id(0);

    if (lid == 0) {
        int sum = 0;
        for (int i = 0; i < bufferSize; i++) {
	        sum += buffer[id + i];
	    }
        float average = sum / bufferSize;
        for (int i = 0; i < bufferSize; i++) {
            buffer[id + i] = average;
        }
    }
}
