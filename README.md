# CS580 Final Project
## How to run the code
Make sure you download all the dependencies in requirements.txt. You could run this in a virtual environment

python -m venv venv_name
source ./venv_name/bin/activate
pip install -r requirements.txt

### Commands
Only run main.py for database and image generation code. You will need to modify your USERNAME and PASSWORD at the top of the main.py file. Parameters can also be changed in there too.

Required arguments:
--db [postgres, mongo, arango]
--op [drop, query, generate]

Examples:
python main.py --db mongo --op generate
python main.py --db arango --op generate
python main.py --db postgres --op generate
python main.py --db postgres --op drop
python main.py --db postgres --op query

### Decoding images
Run decode.py if you want to verify an image loaded in a database is an actual image. You must modify the base64 bytearray string within the file.

python decode_image.py


### Attributions
Much of the generator.py code was borrowed from a Hugging Face Colab notebook.
- https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb#scrollTo=REF_yuHprSa1
- https://huggingface.co/blog/stable_diffusion