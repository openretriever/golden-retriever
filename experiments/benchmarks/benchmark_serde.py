
import time
import numpy as np
import pyarrow as pa
from retriever.rt.backend.dora.serde import serialize_arrow, deserialize_arrow

def benchmark_serde(size_mb):
    # Create a large numpy array
    n_elements = (size_mb * 1024 * 1024) // 8  # float64 = 8 bytes
    data = np.arange(n_elements, dtype=np.float64)
    
    print(f"\n--- Benchmarking size: {size_mb} MB ---")
    
    # --- Current Method (tobytes) ---
    start_time = time.time()
    # Simulate current implementation
    arr_bytes = data.tobytes(order="C")
    arrow_array_current = pa.array([arr_bytes], type=pa.binary())
    ser_time_current = time.time() - start_time
    
    start_time = time.time()
    # Simulate current deserialization
    raw = arrow_array_current[0].as_py()
    restored_current = np.frombuffer(raw, dtype=np.float64).reshape(data.shape)
    deser_time_current = time.time() - start_time
    
    print(f"[Current]   Serialize: {ser_time_current:.6f} s | Deserialize: {deser_time_current:.6f} s")
    
    # --- Proposed Method (pa.array zero-copy) ---
    start_time = time.time()
    # Zero-copy arrow array creation
    # We must ensure it's contiguous, otherwise pa.array might copy.
    # arange is contiguous.
    arrow_array_opt = pa.array(data.reshape(-1)) 
    ser_time_opt = time.time() - start_time
    
    start_time = time.time()
    # Zero-copy deserialization
    restored_opt = arrow_array_opt.to_numpy().reshape(data.shape)
    deser_time_opt = time.time() - start_time
    
    print(f"[Proposed]  Serialize: {ser_time_opt:.6f} s | Deserialize: {deser_time_opt:.6f} s")
    
    # Verify correctness
    assert np.array_equal(data, restored_current)
    assert np.array_equal(data, restored_opt)
    
    # Verify zero-copy nature of proposed method (if possible)
    # arrow_array_current buffers[1] is the data.
    
    print("Data verification successful")

if __name__ == "__main__":
    for size in [10, 100, 500]:
        benchmark_serde(size)
