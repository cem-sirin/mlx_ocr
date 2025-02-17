<div align="center">
    <h1>MLX-OCR</h1>
    <p>
        <b>⚡️ Fast and Efficient OCR Library using Apple MLX Framework 🍎</b>
    </p>
</div>

<br />

## 🚀 Quick Start

Install the library using pip:
```bash
pip install mlx-ocr
```
Then, badabim badabum, you can use the library like this:

```python
from mlx_ocr.models import MLXOCR

ocr = MLXOCR(det_lang="eng", rec_lang="eng")
img = "path/to/image.jpg"
result = ocr(img)
print(result)
```
Check out the [examples](examples) directory for more usage examples!

## 💡 Current Models & Future Models

### Current Models (PP-OCRv3):

Currently, `mlx-ocr` implements the detection and recognition models from the student version of PP-OCRv3. These models are known for their efficiency and good performance.

- **Text Detection:** Implemented and ready to use.
- **Text Recognition:** Implemented and ready to use.

As the PP-OCRv3 language models share the same architecture and training configuration, weights should be broadly compatible across different languages.  Refer to the [PaddleOCR model list](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/models_list.md) for detailed model information.

### Future Model Implementations:

The focus will be on expanding model coverage and algorithmic improvements. Planned model additions include:

- **Angle Classification Model (PP-OCRv3):** To enhance accuracy by correcting text orientation.
- **PP-OCRv4 Models:** Exploring implementation of the next generation PP-OCRv4 models for potential performance gains.
- **Experimentation with other architectures:** Investigating and potentially implementing other state-of-the-art OCR models within the MLX framework.

    
## ✨ Upcoming Features

- [Planned] Beginner-Friendly Fine-tuning Tools: Creating user-friendly tools and guides for fine-tuning models and training new models, making the library more accessible to users with varying levels of expertise.
- [Planned] Enhanced Documentation and Examples: Expanding documentation with more detailed explanations, tutorials, and diverse usage examples to improve user onboarding and understanding.

## 🥲 Disclaimer

Many functionalities are missing, and some parts of the code is not optimized. The models on their own are very performant, but preprocessing and postprocessing needs more work.

People who are used to training models and preparing datasets can easily use the models in this repository. However, I will try to make the library more beginner-friendly in the future, for fine-tuning and training new models.

## 🙌 Contributing

Contributions are welcome and encouraged!  We strive to keep dependencies minimal and appreciate contributions in various forms:

- Model Implementations: Help implement new OCR models within the MLX framework.
- Algorithmic Improvements: Contribute to pre-processing, post-processing, or other algorithmic enhancements.
- Bug Reports: Report any issues or unexpected behavior you encounter.
- Feature Requests: Suggest new features or improvements you'd like to see.
- Documentation: Help improve documentation, tutorials, and examples.
- Code Review: Review and provide feedback on code changes.

The `mlx_ocr/models` directory is intentionally kept lightweight, depending only on `mlx`. We use `pypdfium2` for PDF operations due to its permissive licensing (Apache 2.0 and BSD 3-Clause).

## 🙏 Acknowledgements

This project is inspired by and builds upon the excellent work of the [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) library. Code is adapted and translated (sometimes directly) from PaddleOCR. We are committed to proper citation and will continue to refine attributions.  If you notice any missing citations, please let us know.

We also acknowledge inspiration from the [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) repository for code style and project structure.