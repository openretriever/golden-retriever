
import sys
import numpy as np
from dataclasses import dataclass
import pyarrow as pa

# Adjust path to find retriever
sys.path.append(".pixi/envs/default/lib/python3.11/site-packages")

from retriever.rt.backend.dora import serde

@dataclass
class MyData:
    array: np.ndarray

def test():
    arr = np.zeros(1024, dtype=np.uint8)
    obj = MyData(array=arr)
    
    print(f"Testing serialization of Dataclass with Numpy array...")
    try:
        arrow_arr, meta = serde.serialize_arrow(obj)
        print(f"Metadata type: {meta.get('_type')}")
        if meta.get('_type') == 'pickle':
            print("Confirmed: Falls back to Pickle.")
        elif meta.get('_type') == 'dataclass':
            print("Serialized as Dataclass (JSON). Check if array is inside JSON string?")
            # Inspect values
            print(f"Value: {arrow_arr[0].as_py()}")
        else:
            print(f"Serialized as {meta.get('_type')}")
            
    except Exception as e:
        print(f"Serialization failed: {e}")

if __name__ == "__main__":
    test()
