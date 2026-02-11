import blobconverter

blob_path = blobconverter.from_openvino(
    xml=r"D:\thesis\intel\emotions-recognition-retail-0003\FP16\emotions-recognition-retail-0003.xml",
    bin=r"D:\thesis\intel\emotions-recognition-retail-0003\FP16\emotions-recognition-retail-0003.bin",
    data_type="FP16",
    shaves=6
)
print("Blob saved at:", blob_path)