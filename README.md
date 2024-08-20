### DrSimplify-OCR

Medical Prescription OCR System
<br><br>
This Python script processes medical prescription images to extract text using OCR. It starts by adjusting images for optimal OCR clarity—converting to grayscale and resizing. Text regions are identified and organized into lines using geometric and clustering techniques, based on the dimensions and alignment of text blocks. The Microsoft TrOCR model is employed to decode characters from these preprocessed images. Post-OCR, the script matches recognized text against a database of medications to identify prescribed drugs accurately using fuzzy string matching algorithm. The application requires dependencies including TensorFlow, OpenCV, and the Transformers library, and is structured for straightforward execution with minimal setup.The application is wrapped in a Django framework, providing a straightforward web interface for users to interact with the system on a local server.
