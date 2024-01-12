import os

from modelscope import snapshot_download


def download():
    os.system("git clone https://gitee.com/yzy0612/nltk_data.git  --branch gh-pages && "
              "cd nltk_data && "
              "mv packages/* ./ && "
              "cd tokenizers && "
              "unzip punkt.zip && "
              "cd ../taggers && "
              "unzip averaged_perceptron_tagger.zip")
    os.makedirs('data/model', exist_ok=True)
    snapshot_download('Shanghai_AI_Laboratory/internlm-chat-7b', cache_dir='data/model', revision='v1.0.3')
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.system(
        'huggingface-cli download --resume-download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir data/model/sentence-transformer')