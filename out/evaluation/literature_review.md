# 사람 실험 없는 진동 생성 평가: 원문 확인

조사일: 2026-10-25. 전수 문헌고찰이 아니라 공개 원문의 실험 절을 확인한 조사다.
‘촉각’ 검색에 섞이는 GelSight 센서 이미지 생성은 사람에게 전달하는 진동과 구분했다.

## 자동 평가만 보고한 직접 관련 사례

### Cai et al., IEEE Robotics and Automation Letters 6(4), 7525–7532 (2021)

Visual-Tactile Cross-Modal Data Generation using Residue-Fusion GAN with Feature-Matching and Perceptual Losses.

원문: https://shaoyuca.github.io/mypage/assets/img/Visual_Tactile_Generation.pdf
메타데이터: https://arxiv.org/abs/2107.05468

- 입력/출력: 재질 이미지와 접촉 가속도의 진동 스펙트로그램 사이 양방향 생성.
- 데이터: LMT-108에서 고른 9개 재질. 약하게 짝지은 이미지·스펙트로그램을 증강해 사용.
- 근거 위치: IV Data Preparation, V-A Evaluation Metrics, Table I, VI Discussion.
- 평가: 미리 학습한 DenseNet-121 재질 분류기의 생성 데이터 분류 정확도와 FID.
- 이미지→촉각(V2T) 정확도: pix2pix 40.89%, full model 83.78%.
  FID: 165.60 → 48.40. 촉각→이미지 결과와 혼동하면 안 된다.
- 원문에서 보고한 평가는 자동 분류·분포 비교·제거 실험이며 사람의 체감 평가는 보고하지 않는다.
- 의미: 실제 접촉 진동의 시간–주파수 표현을 대상으로 한 비사용자 평가 선례.
- 한계: 텍스트 입력이 아니다. 분류기가 잘 맞힌다는 사실은 체감 현실감과 다르다.
  평가 분류기는 생성 모델의 조건부 모듈에도 연결되므로 완전히 독립적인 지각 평가기는 아니다.

### Neshov et al., Sensors 26(2):426 (2026)

Visual-to-Tactile Cross-Modal Generation Using a Class-Conditional GAN with Multi-Scale Discriminator and Hybrid Loss.

원문: https://pmc.ncbi.nlm.nih.gov/articles/PMC12845744/
DOI: https://doi.org/10.3390/s26020426

- 입력/출력: 표면 이미지 → 진동 스펙트로그램.
- 데이터: LMT-108의 9개 재질 부분집합.
- 근거 위치: 4 Experiments, Table 4, 5 Discussion의 Absence of Time-Domain Evaluation.
- 평가: LPIPS·FID, baseline 및 제거 실험. Table 4는 10회 실행 평균과 표준편차를 보고한다.
- 사용자 실험을 향후 과제로 명시한다. 시간 영역으로 복원한 진동도 현재 연구에서 평가하지 않았다고 명시한다.
- 의미: 자동 지표만으로 진동 표현 생성 방법을 비교한 확실한 선례.
- 한계: LPIPS·FID는 스펙트로그램을 이미지로 취급한 지표다. 사람의 촉각 품질에 검증된 점수라고 해석하지 않는다.

## 자동 평가만 하지만 목적이 다른 인접 사례

### Wang et al., Synesthesia of Vehicles (2026 arXiv v1)

원문: https://arxiv.org/html/2602.01832v1

- 도로 이미지에서 차량의 도로 유발 tactile excitation을 생성한다.
- IV Experiments and Results: RMSE, FID, 자체 주파수 지표 freq ssim, FFT 및 분류 관련 분석.
- 사람 체감 평가를 보고하지 않는다. 차량 센싱 목적이며 Quest용 진동 콘텐츠 생성과는 다르다.
- 자체 freq ssim은 식과 결과에서 방향 해석이 불명확하므로 그대로 가져오지 않는다.

## 텍스트→진동에서 유용한 자동 평가 사례 (사람 실험도 수행)

### Prompt-to-Touch, ACM TOCHI 33(4), Article 44 (2026)

원문: https://doi.org/10.1145/3819584

- 7 Technical Analysis, 특히 7.3 Expressiveness를 확인했다.
- CLAP으로 텍스트와 생성 **오디오**의 임베딩 cosine similarity를 비교한다.
  의도한 시나리오 설명과 다른 시나리오 설명 사이 구별을 기술적으로 분석한다.
- 오디오 영역에서 가져온 지표의 한계와 표준 촉각 지표 부재를 명시한다.
- 사람 실험도 수행(n=20, n=10). 따라서 ‘사람 실험 없는 논문’으로 분류하면 잘못이다.
- 우리에게 중요한 선례: 텍스트→진동 파이프라인의 기술 분석에 학습된 자동 유사도 지표를 사용했다.
  단, 오디오 CLAP을 HapticGen 진동 WAV에 그대로 적용해 검증된 의미 점수라고 주장할 수 없다.

## 제외/정정

- HapticGen: 32명 사용자 평가를 수행. 자동 평가만의 사례가 아니다.
  https://hapticgen.hcitech.org/
- HapticLDM: 30명 사용자 평가. 자동 평가만의 사례가 아니다.
  https://arxiv.org/html/2605.09971v1
- Scene2Hap의 2025 v1에도 10명 설문 및 10명 촉각 경험 실험이 명시된다.
  일부 문헌의 ‘사용자 실험 없음’ 분류를 재인용하지 않는다.
  https://arxiv.org/html/2504.19611v1
- Audio-to-Tactile Residual U-Net: 공개 초록에 사용자 평가가 명시되어 있어 자동 평가만의 사례로 포함하지 않는다.
- HapticCap의 retrieval 모델과 후속 HapticLLaMA의 caption 생성 모델은 진동 생성기가 아니다.

## 우리 프로젝트에 적용할 설계 (아래는 제안)

1. 독립 benchmark: 횟수, 순서, 강약, 감쇠, 공백을 별도 정답으로 정의한다.
   원래 문장보다 생성 계획을 정답으로 쓰는 현재 평가는 예비 분석으로 유지한다.
2. 고정 데이터셋과 여러 seed에서 baseline·합성·구성 요소 제거를 비교한다.
   동일 문장의 반복은 문장 단위로 집계하고 paired 차이와 신뢰구간을 보고한다.
3. 구조: 충격 검출 F1, 매칭 onset 오차, 반복 간격 오차, 강도 순서 상관.
4. 신호 충실도: 독립 기준 파형이 있는 실험에서만 envelope 오차와 multi-resolution STFT 거리를 측정한다.
   자유 생성은 정답 파형이 유일하지 않으므로 sample MSE만으로 순위를 매기지 않는다.
5. 의미: HapticCap의 고정 retrieval 모델을 검증한 후 보조 지표로 쓸 수 있다.
   단일 진동 모델이 복합 이벤트의 순서를 보존하는지 별도로 검증한다.
6. 지표 타당성: 충격 누락, 순서 변경, 타이밍 이동, 강도 반전 등 통제된 오류에
   점수가 제대로 반응하는지 검사한다. 이것은 사람 지각과의 검증을 대신하지 않지만 구조 점수의 타당성을 강화한다.
7. FID/LPIPS를 추가하려면 비교 데이터, 표본 수, 전처리, encoder를 고정해야 한다.
   영상 encoder의 점수를 촉각 지각 점수로 부르지 않는다.

결론: 사람 실험 없는 진동 생성 평가의 출판 선례는 있다. 확인한 직접 사례는 주로
재질 조건의 스펙트로그램 생성이다. 텍스트→복합 진동의 검증된 범용 자동 채점기는
이 조사에서 확인하지 못했다. 이는 ‘사람 없이 연구할 수 없다’는 뜻이 아니다.
