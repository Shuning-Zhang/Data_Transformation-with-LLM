# Data_Transformation-with-LLM

The testing objective entails evaluating the code generation proficiency of large language models (LLMs), with a specific emphasis on data transformation tasks.

## Installation
```
pip install -r requirements.txt
```

#### Run Prose Api
nevigate to the prose-api folder under model
```
cd Transformation-Text
bash run.sh
```

#### Run Foofah
Build Foofah container
```
docker build -t foofah .
```
Inside Foofah contrainer
```
docker run -p 8080:8080 foofah
bash run.sh
```
Foofah web service will be available at localhost:8080.

#### Run LLMs
nevigate to corresponding LLM folder under models, llama2 and mistral may require GPU for faster running time
```
python ghat_gpt.py
python llama2.py
python mistral.py
```
