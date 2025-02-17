# Module for handling PP-OCR related model loading

I decided to structure the code in a way that functionalities regarding general model loading are handled in utils, and functionalities regarding specific models can be handled in their respective modules. This way, the code is more modular and easier to maintain.

## Possible Issues

### Chinese and Multi-language Detection Model
    
For some reason there is a link to download the student detection model for English, while, for Chinese and multi-language that is not the case. You can only download the distillation model, which is composed of a teacher and two students. This could be annoying for slow internet connections or working with limited memory. By default, I chose the first student model. Feel free to express your opinion on this matter, and we can discuss it. I never used the Chinese or multi-language models, so I don't know if the decisions I explained above are the best ones.