# %% [markdown]
# # Requirements

# %% [markdown]
# - 코랩 환경을 가정하고 필요한 라이브러리를 다운로드 하기 위한 코드입니다.

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
# 필요한 라이브러리 다운로드
!pip install gluonnlp pandas tqdm
!pip install pyproject.toml
!pip install tokenizers
!pip install mxnet
!pip install sentencepiece
!pip install transformers
!pip install torch

# %%
# KoBERT 깃허브 클론
!pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
!pip install git+https://git@github.com/SKTBrain/KoBERT.git@master

# %%
# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
import urllib.request
import os
from tqdm import tqdm
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel


# %% [markdown]
# # Prepare data & tokenizer load

# %% [markdown]
# - 모델 훈련을 위해서 데이터와 토크나이저를 로드하는 공간입니다.

# %%
# 데이터 불러오기
import urllib.request
urllib.request.urlretrieve("https://raw.githubusercontent.com/201803854/Alice/master/deep_learning_data/code_alice.csv", filename="code_alice.csv")
train_data = pd.read_csv('code_alice.csv', names=['document', 'category', 'code_label'], encoding='cp949')

# %%
train_data

# %%
# 카테고리를 숫자로 매핑
mapping = {'category': {'가정폭력':0, '일반절도':1, '살인':2, '성매매/알선':3, '과도노출':4, '무허가주류/담배':5, '무전취식': 6, '보이스피싱':7, '손괴': 8, '강제추행/강간': 9, '범행예고': 10, '시비난동행패소란':11, '일반소음': 12, '적재물낙하': 13, '주취자보호': 14, '미귀가자': 15, '교통서비스': 16, '층간소음': 17, '화재': 18, '가출/실종': 19, '도박': 20, '미성년자고용/출입': 21, '보이스피싱': 22, '공사장소음': 23, '강력 일반폭력': 24, '일반폭력': 25, '분실물신고': 26, '습득신고': 27, '길안내': 28, '전기누전': 29, '수도관파열': 30, '불법주정차': 31, '시설민원': 32, '쓰레기무단투기': 33, '교통사고': 34, '유기': 35, '다중밀집': 36, '일상대화': 37}}
train_data_2 = train_data.replace(mapping)
train_data_2

# %%
train_data_2.dtypes

# %%
# 모델 토크나이저 로드
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")


# %%
# Train, Valid 데이터 분리
from sklearn.model_selection import train_test_split
data = train_data['document']
target = train_data['code_label']
target_ = train_data_2['category']
x_train, x_valid, y_train, y_valid = train_test_split(data, target, test_size=0.1, shuffle=True, random_state=34)
x_train, z_valid, z_train, z_valid = train_test_split(data, target_, test_size=0.1, shuffle=True, random_state=34)


# %%
# y 는 코드분류, z는 사건 유형분류, 주어지는 문장 X_train, text 는 토큰화
X_train_list = x_train.tolist()
X_test_list = x_valid.tolist()

y_train = y_train.tolist()
y_test = y_valid.tolist()

z_train = z_train.tolist()
z_test = z_valid.tolist()

X_train = tokenizer(X_train_list, truncation=True, padding=True)
X_test = tokenizer(X_test_list, truncation=True, padding=True)

# %%
# 데이터셋 형태 준비
import tensorflow as tf
from transformers import TFBertForSequenceClassification
from tensorflow.keras.callbacks import EarlyStopping

train_dataset = tf.data.Dataset.from_tensor_slices((
dict(X_train),
y_train
))
val_dataset = tf.data.Dataset.from_tensor_slices((
dict(X_test),
y_test
))

train_dataset_z = tf.data.Dataset.from_tensor_slices((
dict(X_train),
z_train
))
val_dataset_z = tf.data.Dataset.from_tensor_slices((
dict(X_test),
z_test
))

# %% [markdown]
# # Model Train & Evaluation

# %%
# optimizer 선언
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
optimizer_category = tf.keras.optimizers.legacy.Adam(learning_rate=5e-5)

# %%
# 모델 선언, 코드분류는 0 1 2 3 4 5 => 6개
model = TFBertForSequenceClassification.from_pretrained("klue/bert-base",
num_labels=6, from_pt=True)
model.compile(optimizer=optimizer, loss=model.hf_compute_loss, metrics=['accuracy'
])

# 모델 선언, 카테고리분류는 0부터 37까지 38개
model_category = TFBertForSequenceClassification.from_pretrained("klue/bert-base",
num_labels=38, from_pt=True)
model_category.compile(optimizer=optimizer_category, loss=model.hf_compute_loss, metrics=['accuracy'
])

# %%
# 코드분류 모델 훈련
early_stopping = EarlyStopping(
monitor="val_accuracy",
min_delta=0.001,
patience=2)

model.fit(
train_dataset.shuffle(10000).batch(32), epochs=5, batch_size=32,
validation_data = val_dataset.shuffle(10000).batch(32),
callbacks = [early_stopping]
)

# %%
# 카테고리 분류 모델 훈련
early_stopping = EarlyStopping(
monitor="val_accuracy",
min_delta=0.001,
patience=2)

model_category.fit(
train_dataset_z.shuffle(10000).batch(32), epochs=5, batch_size=32,
validation_data = val_dataset_z.shuffle(10000).batch(32),
callbacks = [early_stopping]
)

# %%
# 코드분류 모델 평가
model.evaluate(val_dataset.batch(1024))

# %%
# 카테고리 분류 모델 평가
model_category.evaluate(val_dataset_z.batch(1024))

# %% [markdown]
# # Hugginface push
# - 허깅페이스 푸쉬를 위한 코드입니다.
# - 이런식으로 훈련이 끝난 모델을 kyungmin011029/~ 에 푸쉬하여 저장했습니다.
# - 테스트 해볼 시에는 푸쉬할 필요 없시 아래 Prediction load my model 셀부터 실행하면 됩니다.

# %%
# 허깅페이스 로그인
!pip install huggingface_hub transformers
from huggingface_hub import notebook_login

notebook_login()
#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

# %%
# 허깅페이스 레포지토리 생성, 푸쉬
from huggingface_hub import notebook_login, create_repo
create_repo("kyungmin011029/code_alice", private=False)
create_repo("kyungmin011029/category_alice", private=False)

from transformers import AutoModel
from transformers import AutoTokenizer


# %%
# Huggingface Access Token
ACCESS_TOKEN = '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'

# Upload to Huggingface
model.push_to_hub('kyungmin011029/code_alice', use_temp_dir=True, use_auth_token=ACCESS_TOKEN)
tokenizer.push_to_hub('kyungmin011029/code_alice', use_temp_dir=True, use_auth_token=ACCESS_TOKEN)

# %%
# Huggingface Access Token
ACCESS_TOKEN = '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'


# Upload to Huggingface
model_category.push_to_hub('kyungmin011029/category_alice', use_temp_dir=True, use_auth_token=ACCESS_TOKEN)
tokenizer.push_to_hub('kyungmin011029/category_alice', use_temp_dir=True, use_auth_token=ACCESS_TOKEN)

# %% [markdown]
# # Prediction (Load my model)

# %%
from transformers import TextClassificationPipeline

# 푸쉬한 모델 가져오기 (코드분류)
loaded_tokenizer = BertTokenizerFast.from_pretrained('kyungmin011029/code_alice')
loaded_model = TFBertForSequenceClassification.from_pretrained('kyungmin011029/code_alice', use_auth_token=True)

text_classifier = TextClassificationPipeline(
tokenizer=loaded_tokenizer,
model=loaded_model,
framework='tf',
return_all_scores=True
)

# 푸쉬한 모델 가져오기 (사건유형분류)
loaded_tokenizer_category = BertTokenizerFast.from_pretrained('kyungmin011029/category_alice')
loaded_model_category = TFBertForSequenceClassification.from_pretrained('kyungmin011029/category_alice', use_auth_token=True)

text_classifier_category = TextClassificationPipeline(
tokenizer=loaded_tokenizer_category,
model=loaded_model_category,
framework='tf',
return_all_scores=True
)

# %% [markdown]
# # Prediction
# - 여기서 예측이 가능합니다.
# - code_classifier 는 코드와 카테고리 모두 예측하도록 짜여진 함수입니다.
# - 코드, 카테고리 따로도 예측할 수 있습니다.

# %%
def code_classifier():
  text_input = input('궁금한 상황은?')
  result = text_classifier(text_input)[0]
  category = text_classifier_category(text_input)[0]
  #result[0]['score']
  max_prob = result[0]['score']
  check_code = 0
  for i in range(6):
    if max_prob < result[i]['score']:
      check_code = i
      max_prob = result[i]['score']
  max_prob *= 100

  category_prob = category[0]['score']

  check = 0
  for i in range(0, 38):
    if category_prob < category[i]['score']:
      category_prob = category[i]['score']
      check = i
  mapping_ = {'가정폭력':0, '일반절도':1, '살인':2, '성매매/알선':3, '과도노출':4, '무허가주류/담배':5, '무전취식': 6, '보이스피싱':7, '손괴': 8, '강제추행/강간': 9, '범행예고': 10, '시비난동행패소란':11, '일반소음': 12, '적재물낙하': 13, '주취자보호': 14, '미귀가자': 15, '교통서비스': 16, '층간소음': 17, '화재': 18, '가출/실종': 19, '도박': 20, '미성년자고용/출입': 21, '보이스피싱': 22, '공사장소음': 23, '강력 일반폭력': 24, '일반폭력': 25, '분실물신고': 26, '습득신고': 27, '길안내': 28, '전기누전': 29, '수도관파열': 30, '불법주정차': 31, '시설민원': 32, '쓰레기무단투기': 33, '교통사고': 34, '유기': 35, '다중밀집': 36, '일상대화': 37}
  map = {v:k for k,v in mapping_.items()}
  get = map.get(check)

  return "궁금한 상황의 예측된 코드번호 분류는 {}이며, 예측 확률은 {}입니다. 해당 사건의 카테고리는 {}입니다.".format(check_code, max_prob, get)

#print("궁금한 상황의 예측된 코드번호 분류는 {}이며, 예측 확률은 {}입니다.".format(result['label'], result['score']))

# %%
# 여기서 카테고리만 따로 테스트할 수 있습니다.
text_classifier_category('한 남자가 옥상에서 여자를 묶고 죽였어요')[0]


# %%
# 여기서 코드분류만 따로 테스트해볼 수 있습니다.
text_classifier('외대앞역이에요')[0]

# %%



