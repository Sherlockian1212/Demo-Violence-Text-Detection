# import keras
# import tensorflow as tf
# print(tf.__version__)
# print(keras.__version__)
from transformers import TFBertModel

# # Đường dẫn đến thư mục lưu cache
# cache_dir = './my_model_cache'

# # Tải mô hình BERT và lưu vào thư mục cache đã chỉ định
# bert_model = TFBertModel.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
# Đường dẫn đến thư mục lưu cache
cache_dir = './my_model_cache/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594'
# Tải lại mô hình từ thư mục cache đã lưu
bert_model = TFBertModel.from_pretrained(cache_dir)
