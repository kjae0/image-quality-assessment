각 이미지의 mos 예측을 위한 소스코드입니다.
재현을 위한 실행 순서는 아래와 같습니다.
각 .py 파일에서 변수 설정을 한 후 작동시켜야 오류가 발생하지 않습니다. 

1. feature extraction
- autogluon 학습을 위해 frozen backbone에서 feature들을 추출하는 과정입니다.
- 각 extract_features_{model_name}_{flip 여부}.py를 실행시키면 됩니다.

2. train autogluon models
- step 1에서 추출한 feature로 autogluon 모델을 학습시킵니다.
- 4가지 모델(vit, nfnet, swin1k, swin22k)로부터 추출된 feature들을 개별적으로 학습시킵니다. (총 8 모델)
- glu_run_{model_name}.sh를 실행시키면 됩니다. (각 .sh 파일의 경로설정을 해줘야합니다.)

3. inference
- step 2에서 학습시킨 autogluon 모델을 활용하여 예측값을 얻습니다.
- 8가지 모델 * (원본, flip된 버전)의 test dataset을 활용하여 16가지의 예측값을 얻습니다.
- autogluon_inference.py를 실행시키면 됩니다. (16가지 output값을 담은 파일이 .pkl로 저장됩니다.)
